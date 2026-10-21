# -*- coding: utf-8 -*-
"""
Bayesian network for PVRV (PV-101) opening - one-hour slice formulation

Nodes (within one hourly slice)
  evidence   Qcls (gas production class)  Fcls (holder fill class)  DN (day/night)  W (winter)
  initiating event I in {none, stop (downstream stop), demand (flare start demand), foam (foam blockage),
                         block (condensate / freezing / XV blockage), wv (weight-loaded vent valve stuck)}
  D (stop duration)  F (flare works on demand)  R (operator response time)  B (foam fouls the PVRV)  L (ratchet reached within the latent period)
  Open (opening)  Vent (vented volume class)
CPTs are built from PARAMS (with sources) and the physical functions (physics.py). Inference is exact enumeration per slice.
"""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics as ph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
ADM1_CSV = os.path.join(DATA_DIR, 'ADM1_hourly_1to9_Q200_6000m3_ffOTS1.csv')
KMA_CSV = os.path.join(DATA_DIR, 'kma_asos138_pohang_hourly_pressure_kPa.csv')

# ----------------------------------------------------------------------------- parameters and sources
PARAMS = {
    'lam_stop':      dict(v=12.0, unit='per year', src='Assumption. Qualitative basis: Reinelt 2016 (many CHP stops in 106 days), Korean experience of frequent faults of gas engines / steam users'),
    'lam_power':     dict(v=0.1,  unit='per year', src='KEPCO interruptions per customer 0.08-0.13 per year (2025 statistics, released via a National Assembly office, Segye Ilbo 2026-07-14)'),
    'D_dist':        dict(v={0.5: 0.45, 2.0: 0.35, 6.0: 0.15, 24.0: 0.05}, unit='h: probability', src='Assumption. Durations in ARIA 47807 (1.5 h), 49983, 42739 (30 min) as reference'),
    'pfd_flare':     dict(v=0.05, unit='probability of failure on demand', src='Assumption. Mid value of the IEC 61511 SIL 1 band (0.01-0.1); failure modes (ignition, flame detection, freezing) after Travnicek'),
    'lam_foam':      dict(v=2.0,  unit='per year', src='Kougias 2014 (16 Danish plants, up to 3 per year), Moeller 2012 (7 of 15 plants regularly)'),
    'p_block_foam':  dict(v=0.3,  unit='probability', src='Assumption. EA 2019 incident review (foam blocking PRVs and gas lines)'),
    'p_pv_fouled':   dict(v=0.1,  unit='probability', src='Assumption (v4.1: 0.3 -> 0.1, the same value as the QRA seal-failure chain). EA 2019 (2012-06 foam blocked the PRV and the roof ruptured; 2013-01 floating layer blocked the PRV). Sensitivity 0.03-0.3'),
    'lam_cond':      dict(v=0.5,  unit='per year', src='Assumption. Condensate pot management failure'),
    'lam_freeze':    dict(v=0.2,  unit='per year (Dec-Feb only)', src='Assumption. ARIA 47807 / 42739 (isolation by freezing, frozen valve)'),
    'lam_xv':        dict(v=0.1,  unit='per year', src='Assumption. Shut-off valve mis-operation / interlock isolation (ARIA 49983)'),
    'lam_wv':        dict(v=0.1,  unit='per year', src='Assumption. Weight-loaded lid stuck (corrosion, freezing). ARIA 42739 (frozen relief valve)'),
    'latent_wv_h':   dict(v=168,  unit='h', src='Assumption. Weekly walk-down inspection'),
    'R_day':         dict(v={0.5: 0.6, 2.0: 0.3, 8.0: 0.1}, unit='h: probability', src='Assumption. Attended operation during the day'),
    'R_night':       dict(v={0.5: 0.2, 2.0: 0.4, 8.0: 0.4}, unit='h: probability', src='Assumption. Reinelt 2016: before monitoring the operators were unaware of the releases'),
    'Q_use':         dict(v=60.0, unit='m3/h', src='User setting (constant consumption)'),
    'fill_series':   dict(v='fill_hourly_use60.csv', unit='', src='Operating model (constant consumption 60 + proportional flare 150, 70-85 %)'),
    'demand_series': dict(v='flare_demands_use60.csv', unit='', src='Operating model: 641 flare demands per year'),
    'chatter_per_h': dict(v=138.0, unit='per h', src='1-s model, minimum stable opening 10 %'),
    # ---- operating regime (v4: on/off operation)
    'f_on':          dict(v=0.85, unit='true fill fraction', src='Flare start fill fraction (on/off operation: 1.0 / 0.8 / 0.6)'),
    't_retry_h':     dict(v=0.0,  unit='h', src='Automatic retry interval after an ignition failure. 0 = no automatic retry (operator response only). 0.5 matches the 1-s model'),
    'n_retry':       dict(v=4,    unit='count', src='Maximum number of automatic retries; if all fail the event passes to the operator response (R)'),
    'p_hard':        dict(v=0.0,  unit='share of failures', src='Assumption. Share of failures that a retry cannot clear (flame detector, pilot gas, freezing, cf. Travnicek) -> operator response only. Sensitivity value 0.3'),
    'pressure_backup': dict(v=True, unit='', src='Whether a pressure-based flare start (PSH) exists. With fill-level start only, False -> no backup in the stuck-vent-valve branch'),
    # ---- optional nodes (defaults have no effect)
    'level_bias':    dict(v={0.0: 1.0}, unit='fill error: probability', src='Assumption. Ultrasonic level indicator error (inner membrane shape). Sensitivity: {-0.05: .25, 0: .5, +0.05: .25}'),
    'p_pv_stuck':    dict(v=0.0048, unit='probability of failure per opening demand (one valve)', src='NOG-070 (2020) Table A.2.1, PSV: lambda_DU 1.1e-6 /h (fail to open before test pressure), annual proof test: PFD = lambda tau / 2 = 4.8e-3. Sensitivities: 2.2e-6 /h (all DU modes), dirty service x5 (foam, sulphur, condensate)'),
    'beta_pv':       dict(v=0.1,  unit='common-cause factor', src='IEC 61508-6 beta for two identical valves on a shared header (1oo2: PFD = beta PFD + PFD^2); stop / demand / wv branches. Foam / block branches isolate one digester -> one valve'),
    'p_reseat':      dict(v=0.0, unit='probability of reseat failure per opening episode', src='Assumption. Seat fouling. Sensitivity value 0.05'),
    'Q_leak':        dict(v=20.0, unit='m3/h', src='Assumption. Leak rate after a reseat failure'),
    't_leak_h':      dict(v=12.0, unit='h', src='Assumption. Mean time until a leak is found and fixed'),
}
DAY = range(8, 20)   # daytime 08-20 h

def load_inputs(base=None, prm=None):
    prm = prm or PARAMS
    d = pd.read_csv(ADM1_CSV, parse_dates=['datetime_interval_end'])
    q = d['Q_gas_wet_m3_d_20C'].to_numpy() / 24.0
    yCH4 = d['CH4_wet_pct'].to_numpy() / 100
    k = pd.read_csv(KMA_CSV); k.columns = ['dt', 'p']; k['dt'] = pd.to_datetime(k['dt'])
    p = k.set_index('dt')['p'].reindex(d['datetime_interval_end']).ffill().bfill().to_numpy()
    f = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), prm['fill_series']['v']))['fill'].to_numpy()
    fill = np.concatenate([f, [f[-1]]])
    dem = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), prm['demand_series']['v']))['t_s'].to_numpy()
    demand_h = np.zeros(len(q)); np.add.at(demand_h, np.minimum((dem // 3600).astype(int), len(q) - 1), 1.0)   # flare start demands per hour
    t = d['datetime_interval_end']
    return dict(t=t, q=q, yCH4=yCH4, p=p, fill=fill, demand=demand_h,
                day=t.dt.hour.isin(DAY).to_numpy(), winter=t.dt.month.isin([12, 1, 2]).to_numpy())

# ----------------------------------------------------------------------------- per-slice inference (exact enumeration)
def slice_inference(case, q, fill, p_atm, demand, day, winter, prm, wv_prob):
    """One hourly slice: per initiating event (P(I), P(open | I), expected vented m3, expected open duration h, list of intervals [(weight, start h, end h)])"""
    out = {}
    use = prm['Q_use']['v']
    R = prm['R_day']['v'] if day else prm['R_night']['v']
    pfd = prm['pfd_flare']['v']
    bias = prm['level_bias']['v']
    f_on = prm['f_on']['v']
    # "recovery time" distribution after an ignition failure: whichever comes first of the automatic retries (interval t_retry, at most n_retry, each succeeding with 1-pfd) and the operator response R.
    #   t_retry = 0 means no automatic retry -> operator response only (same as v3)
    t_rt, n_rt, ph_ = prm['t_retry_h']['v'], int(prm['n_retry']['v']), prm['p_hard']['v']
    #   a share (1-p_hard) of the failures is transient (cleared by a retry); p_hard are hard faults (retries useless -> operator only)
    retry = ([(k * t_rt, (1 - ph_) * (1 - pfd) * pfd ** (k - 1)) for k in range(1, n_rt + 1)] + [(np.inf, (1 - ph_) * pfd ** n_rt + ph_)]) if t_rt > 0 else [(np.inf, 1.0)]
    recov = {}                                   # recovery time -> probability (R combined with the retries)
    recov_op = {}                                # ... of which the operator arrives first (used by the QRA for responder exposure)
    for r, pr in R.items():
        for tk, pk in retry:
            e = min(r, tk); recov[e] = recov.get(e, 0.0) + pr * pk
            if r < tk: recov_op[e] = recov_op.get(e, 0.0) + pr * pk
            elif r == tk: recov_op[e] = recov_op.get(e, 0.0) + 0.5 * pr * pk     # simultaneous: half the weight
    # --- downstream stop: all production goes to the holder -> flare demand at indicated f_on -> failure (pfd) -> opening if the full + rigid phase is passed before recovery
    p_stop = (prm['lam_stop']['v'] + prm['lam_power']['v']) / 8760
    t_open = float(ph.t_full_h(fill, q, p_atm, case) + ph.t_rigid_h(case, q))
    po = 0.0; vent = 0.0; dur = 0.0; iv = []
    for b, pb in bias.items():
        fon = min(f_on + b, 0.999)
        t_on = (fon - fill) * ph.V_MAX * (p_atm + ph.P_OP[case]) / ph.P_REF / q if fill < fon else 0.0
        for D, pD in prm['D_dist']['v'].items():
            for e, pe in recov.items():
                end = min(D, t_on + e)                # downstream recovery or flare recovery, whichever is first
                if t_open < end:
                    w = pb * pD * pe * pfd
                    op = (recov_op.get(e, 0.0) / pe) if (t_on + e <= D and pe > 0) else 0.0     # share of this weight where the operator finds the valve venting
                    po += w; vent += w * (end - t_open) * q; dur += w * (end - t_open); iv.append((w, t_open, end, op))
    out['stop'] = (p_stop, po, vent, dur, iv)
    # --- flare demand (indicated f_on = true fill f_on + bias): the consumer keeps taking 60 -> only the surplus (q - use) raises the holder and the pressure
    if demand > 0 and q > use:
        po = vent = dur = 0.0; iv = []
        for b, pb in bias.items():
            f0 = min(f_on + b, 0.999)
            t_o = float(ph.t_full_h(f0, q - use, p_atm, case) + ph.t_rigid_h(case, q - use))
            for e, pe in recov.items():
                if t_o < e:
                    w = pb * pe * pfd; po += w; vent += w * (e - t_o) * (q - use); dur += w * (e - t_o); iv.append((w, t_o, e, recov_op.get(e, 0.0) / pe if pe > 0 else 0.0))
        out['demand'] = (float(demand), po, vent, dur, iv)
    else:
        out['demand'] = (0.0, 0.0, 0.0, 0.0, [])
    # --- outlet of one digester blocked (foam / other)
    t_o = float(ph.t_iso_h(case, q / 2))
    po = vent = dur = 0.0; iv = []
    for r, pr in R.items():
        if t_o < r:
            po += pr; vent += pr * (r - t_o) * q / 2; dur += pr * (r - t_o); iv.append((pr, t_o, r, 1.0))       # ends with the operator at the valve
    p_foam = prm['lam_foam']['v'] * prm['p_block_foam']['v'] / 8760
    nf = 1 - prm['p_pv_fouled']['v']
    out['foam'] = (p_foam, po * nf, vent * nf, dur * nf, [(w * nf, a, b, op) for (w, a, b, op) in iv])
    p_blk = (prm['lam_cond']['v'] + prm['lam_xv']['v']) / 8760 + (prm['lam_freeze']['v'] / (90 * 24) if winter else 0.0)
    out['block'] = (p_blk, po, vent, dur, iv)
    # --- weight-loaded vent valve stuck (atmospheric ratchet within the latent period; the PVRV opens only if the pressure backup of the flare fails)
    p_wv = prm['lam_wv']['v'] / 8760
    L = prm['latent_wv_h']['v']
    bk = pfd if prm['pressure_backup']['v'] else 1.0   # without a pressure-based start (fill-level start only) there is no backup against the ratchet
    iv = [(bk * wv_prob[0], L - wv_prob[2], L, 1.0)] if wv_prob[0] > 0 else []          # found at the next inspection
    out['wv'] = (p_wv, bk * wv_prob[0], bk * wv_prob[1], bk * wv_prob[2], iv)
    return out

def run_year(case, inp, prm=PARAMS, occupancy=False):
    n = len(inp['q'])
    occ = np.zeros(n + 400); occ_k = {k: np.zeros(n + 400) for k in ('stop', 'demand', 'foam', 'block', 'wv')}
    to, vent = ph.wv_stuck_ratchet(inp['p'], case, latent_h=int(prm['latent_wv_h']['v']))
    reach = ~np.isnan(to)
    rows = []
    for h in range(n):
        j = min(h, len(to) - 1)
        wvp = (float(reach[j]), float(vent[j]), float(np.nan_to_num(prm['latent_wv_h']['v'] - to[j], nan=0.0)) if reach[j] else 0.0)
        o = slice_inference(case, inp['q'][h], inp['fill'][h], inp['p'][h], inp['demand'][h], inp['day'][h], inp['winter'][h], prm, wvp)
        r = dict(h=h, q=inp['q'][h], fill=inp['fill'][h])
        for k, (pI, po, ve, du, iv) in o.items():
            r[f'p_{k}'] = pI * po; r[f'v_{k}'] = pI * ve; r[f'd_{k}'] = pI * du
            r[f'o_{k}'] = pI * sum(w * (it[3] if len(it) > 3 else 0.0) for it in iv for w in [it[0]])   # P(operator arrives while the valve vents)
            if occupancy:
                sk = 0.0
                for it in iv:                        # distribute the open interval [h+a, h+b] over hourly bins -> probability of being open at that hour
                    w, a, b = it[0], it[1], it[2]; wa = pI * w
                    for bn_ in range(int(np.floor(h + a)), int(np.floor(h + b)) + 1):
                        ov = min(h + b, bn_ + 1) - max(h + a, bn_)
                        if ov > 0:
                            occ[bn_] += wa * ov; occ_k[k][bn_] += wa * ov
        rows.append(r)
    df = pd.DataFrame(rows)
    df['p_open'] = df[[c for c in df.columns if c.startswith('p_')]].sum(axis=1)
    df['v_open'] = df[[c for c in df.columns if c.startswith('v_')]].sum(axis=1)
    df['d_open'] = df[[c for c in df.columns if c.startswith('d_')]].sum(axis=1)
    df['ch4_kg'] = df['v_open'] * inp['yCH4'][:n] * 0.667
    if occupancy:
        df['p_state_open'] = occ[:n]            # probability that the PVRV is open at that hour
        for k in occ_k: df[f's_{k}'] = occ_k[k][:n]   # ... by initiating event
    return df

def pv_pfd(prm=PARAMS):
    """PFD of the relief on demand: shared header (two valves, 1oo2 with common cause) and an isolated digester (one valve)"""
    p = prm['p_pv_stuck']['v']; b = prm['beta_pv']['v']
    return dict(shared=b * p + ((1 - b) * p) ** 2, single=p)

def pv_overpressure(df, prm=PARAMS):
    """overpressure events per year because the PVRV does not open when it should (hardware), from the hourly opening demands"""
    f = pv_pfd(prm)
    return float(df[['p_stop', 'p_demand', 'p_wv']].sum().sum() * f['shared'] + df[['p_foam', 'p_block']].sum().sum() * f['single'])

def summarize(df, prm=PARAMS):
    s = {}
    for k in ('stop', 'demand', 'foam', 'block', 'wv'):
        s[k] = dict(openings_per_yr=df[f'p_{k}'].sum(), vent_m3_per_yr=df[f'v_{k}'].sum(), open_h_per_yr=df[f'd_{k}'].sum())
    tot = df['p_open'].sum()
    s['total'] = dict(openings_per_yr=tot, vent_m3_per_yr=df['v_open'].sum(), open_h_per_yr=df['d_open'].sum(),
                      ch4_kg_per_yr=df['ch4_kg'].sum(), P_at_least_one_per_yr=1 - np.exp(-tot),
                      chatter_cycles_per_yr=df['d_open'].sum() * prm['chatter_per_h']['v'],
                      pv_stuck_overpressure_per_yr=pv_overpressure(df, prm),
                      pv_fouled_overpressure_per_yr=prm['lam_foam']['v'] * prm['p_block_foam']['v'] * prm['p_pv_fouled']['v'],
                      reseat_fail_per_yr=tot * prm['p_reseat']['v'],
                      reseat_leak_m3_per_yr=tot * prm['p_reseat']['v'] * prm['Q_leak']['v'] * prm['t_leak_h']['v'])
    return s

# ----------------------------------------------------------------------------- CPT extraction by class (BN table form)
def cpt_tables(case, inp, df, prm=None):
    """Aggregate P(Open | I, Qcls, Fcls, DN) from the slice results (conditional on the initiating event having occurred)"""
    prm = prm or PARAMS
    qcls = pd.cut(inp['q'][:len(df)], [0, 60, 90, 999], labels=['low', 'mid', 'high'])
    fcls = pd.cut(inp['fill'][:len(df)], [-0.01, 0.3, 0.7, 0.85, 1.01], labels=['low', 'mid', 'high', 'full'])
    dn = np.where(inp['day'][:len(df)], 'day', 'night')
    pI = {'stop': (prm['lam_stop']['v'] + prm['lam_power']['v']) / 8760,
          'foam': prm['lam_foam']['v'] * prm['p_block_foam']['v'] / 8760,
          'wv': prm['lam_wv']['v'] / 8760}
    rows = []
    for k, p in pI.items():
        g = pd.DataFrame({'Q': qcls, 'F': fcls, 'DN': dn, 'po': df[f'p_{k}'] / p, 'v': df[f'v_{k}'] / p}).groupby(['Q', 'F', 'DN'], observed=True).agg(n=('po', 'size'), P_open=('po', 'mean'), E_vent_m3=('v', 'mean')).reset_index()
        g.insert(0, 'I', k); rows.append(g)
    dem = df['p_demand'] > 0
    if dem.any():
        g = pd.DataFrame({'Q': qcls[dem], 'DN': dn[dem], 'po': df.loc[dem, 'p_demand'], 'v': df.loc[dem, 'v_demand']}).groupby(['Q', 'DN'], observed=True).agg(n=('po', 'size'), P_open=('po', 'mean'), E_vent_m3=('v', 'mean')).reset_index()
        g.insert(0, 'I', 'demand'); g.insert(2, 'F', f"at f_on({prm['f_on']['v']})"); rows.append(g)
    return pd.concat(rows, ignore_index=True)

if __name__ == '__main__':
    inp = load_inputs()
    os.makedirs('out', exist_ok=True)
    allsum = {}
    for case in (1, 2, 3):
        df = run_year(case, inp, occupancy=True)        # occupancy=True adds the state probability p_state_open
        df.to_csv(f'out/bn_hourly_case{case}.csv.gz', index=False, float_format='%.6g')
        s = summarize(df); allsum[case] = s
        cpt = cpt_tables(case, inp, df); cpt.to_csv(f'out/bn_cpt_case{case}.csv', index=False, float_format='%.4f')
        print(f'Case {case}:', json.dumps({k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in s.items()}, ensure_ascii=False))
    json.dump(allsum, open('out/bn_summary.json', 'w'), indent=1, ensure_ascii=False, default=float)
    json.dump(PARAMS, open('out/bn_params.json', 'w'), indent=1, ensure_ascii=False, default=str)
