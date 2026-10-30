# -*- coding: utf-8 -*-
"""
Dynamic QRA, stage 3: event tree after "PVRV opens" (or fails to open) and the time-resolved risk integration.

Frequency side  : the hourly BN results (upstream/bayesian_network/pv_bn.py, v4.3, pinned from biogas-pvrv-BN) give, per hour h and initiating event k,
                    p_k(h)  probability that an event starting in hour h opens the valve
                    o_k(h)  ... and that the operator reaches the valve while it is still venting
                    s_k(h)  probability that the valve is open during hour h (state probability)
                    d_k(h)  expected open duration (h)
Consequence side: near-field plume on the digester roof (consequence.py) for the weather of that hour (KMA ASOS 138),
                  flash fire (person inside the LFL plume at ignition) and H2S (RIVM 2018 probit) at the H2S content of that hour.
Exposure        : the operator who responds to the event and goes up to the valve, and the routine work on the roof (2 h/week).
Overpressure    : a PVRV that does not relieve - (a) fouled by foam (BN p_pv_fouled) or (b) hardware fail-to-open (NOG-070, 1oo2 with
                  common cause on the shared header, one valve for an isolated digester) - starts a race between the pressure rise
                  (rate from q(h)) and the operator who must open a manual vent on the roof before the seal lets go.
Risk measures   : PLL over everyone who responds or works on the roof, IR of the most exposed individual, LSIR 3 m from the vent,
                  annual CH4 emission; annual, monthly and hour-of-day profiles; by effect (H2S / flash fire / mechanical).
Static QRA      : the same scenarios and annual event numbers with every time-dependent factor (weather, day/night exposure and
                  response, q, H2S) replaced by its annual mean -> the difference to the dynamic result is the time-correlation effect.
Cases           : Case 1 (4.0 kPag) and Case 3 (2.0 kPag) x flare start 100 % / 80 %  (+ the generic-frequency QRA for comparison)
"""
import os, sys, json, copy, math
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'upstream', 'bayesian_network')); sys.path.insert(0, HERE)
import pv_bn as bn
import consequence as cq
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- QRA parameters (sources / assumptions)
QRA = {
    'weather_mode':  dict(v='hourly', src="'hourly': KMA ASOS 138 record, Turner stability class and vent-height wind per hour (met_stability.py); 'mix': the day/night mix below"),
    'weather_day':   dict(v={'D5': 0.5, 'D3': 0.3, 'calm': 0.2}, src='Former assumption, kept for weather_mode = mix: daytime mix of D / 5 m/s, D / 3 m/s and calm'),
    'weather_night': dict(v={'F2': 0.6, 'D3': 0.2, 'calm': 0.2}, src='Assumption: night-time mix of F / 2 m/s, D / 3 m/s and calm'),
    'p_ign':         dict(v=0.02, src='TNO Purple Book (CPR 18E) table 4.3: continuous release of flammable gas < 10 kg/s, immediate + delayed ignition'),
    'p_ign_worker':  dict(v=0.002, src='Assumption: ignition source brought by the worker at the valve (tools, torch, static) per visit; hot-work rules apply'),
    'p_fat_flash_small': dict(v=0.10, src='Assumption: fatality given ignition while inside the sub-metre plume at the vent (face/hand burns likely, death unlikely); TNO uses 1.0 for engulfment in large clouds'),
    'p_fat_flash_R5': dict(v=1.0, src='TNO Green Book: fatality 1.0 inside the flash-fire envelope of the 30-s depressurisation cloud (LFL 6-13 m)'),
    'p_zone':        dict(v=0.10, src='Assumption from the plume geometry (radius 0.1-0.4 m at the LFL distance): probability that a person at 1 m from the vent is inside the flammable plume'),
    'p_zone_R5':     dict(v=0.50, src='Assumption: the 30-s depressurisation cloud (LFL 6-13 m) covers most of the roof'),
    'p_near_valve_routine': dict(v=0.3, src='Assumption: share of the routine time on the roof spent within a few metres of the PVRV (the valve is one of the items of the round)'),
    't_at_valve_min': dict(v=5.0, src='Assumption: minutes a responding operator spends at a venting valve'),
    't_fix_min':     dict(v=15.0, src='Assumption: minutes from arrival until the venting is stopped (flare restart, bypass, clearing the blockage)'),
    'p_in_plume_h2s': dict(v=0.02, src='Assumption: probability that the visit involves a prolonged stay inside the H2S plume (downwind of the vent, leaning over it) despite the personal H2S alarm (10 ppm) that plant staff carry; 0.1-0.3 without a detector (tornado)'),
    'lsir_distance_m': dict(v=3.0, src='Location-specific IR evaluated for a permanent presence 3 m from the vent (roof walkway)'),
    # ---- detection -> diagnosis -> action, per initiating event (what the operator sees, what it means, where they go)
    'detection':     dict(v={'demand': 'flare controller: ignition-failure alarm within ~1 min', 'stop': 'consumer trip alarm, immediate',
                            'foam': 'PAH on PT-101 of that digester (set 4.8 kPag) ~4 min after the blockage, 1-2 min before the valve opens', 'block': 'same as foam',
                            'wv': 'no direct signal; holder pressure trend if a transmitter exists, otherwise the weekly round', 'R5': 'PT-101 above the PVRV setting and still rising = the valve is not relieving'},
                          src='Plant instrumentation (ISO-BG-001 REV.1): PT-101A/B, holder level, flare and consumer status. The PVRV itself has no position switch and there is no gas detector at the vent'),
    'action':        dict(v={'demand': 'control-room restart (same as the automatic retry); if it fails, to the FLARE, not the digester roof', 'stop': 'manual flare start from the control room',
                            'foam': 'field check of the digester: sight glass, condensate pot, XV, PVRV - all on the roof', 'block': 'same as foam',
                            'wv': 'inspection at the holder; digester roof only if PT-101 is high', 'R5': 'stop feeding and open the manual vent ON THE ROOF within the 27 min to seal failure'},
                          src='Assumption'),
    'p_roof':        dict(v={'demand': (0.10, 0.05), 'stop': (0.10, 0.05), 'foam': (0.70, 0.50), 'block': (0.70, 0.50), 'wv': (0.30, 0.20)},
                          src='Assumption (day, night): probability that the response puts the operator on the digester roof while the valve is still venting, following the action column above'),
    'n_responders':  dict(v=3, src='Assumption: one field operator per shift, three shifts; a given individual answers one third of the alarms. IR_worker = PLL_resp / 3 + routine (the 2 h/week are assigned to one person)'),
    'control_room':  dict(v='24 h, three shifts', src='Assumption; the night response distribution R_night (BN) applies 20-08 h'),
    'PAH_kPag':      dict(v=4.8, src='Assumption: high-pressure alarm on PT-101 at 4.8 kPag (0.25 kPa below the PVRV opening pressure 5.05)'),
    'routine_h_per_week': dict(v=2.0, src='User: about 2 h/week of regular work on the digester roofs, in the day shift (08-20, 7 days)'),
    'p_fouled_factor': dict(v=1.0, src='Multiplier on the BN parameter p_pv_fouled (0.1, single source for the BN opening branch and the seal-failure chain); design option B uses 0.1 (common-cause fouling of an independent second relief)'),
    'second_relief': dict(v=False, src='Design option B: a second relief device on its own nozzle of each digester -> hardware fail-to-open of an isolated digester becomes 1oo2 (beta PFD + PFD^2)'),
    'pv_dirty_factor': dict(v=1.0, src='Multiplier on the NOG-070 fail-to-open PFD for dirty service (foam, sulphur, condensate); sensitivity 5'),
    'p_injury_seal': dict(v=0.10, src='Assumption: fatality for a person on the roof when the seal / manhole lets go at 10 kPag (blow-out, fall from the roof edge, engulfment); the flash fire of the 30-s cloud is counted separately (p_zone_R5, p_fat_flash_R5)'),
    'p_vent_fouled': dict(v=0.3, src='Assumption: probability that the manual vent on the same roof is fouled by the same foam as the PVRV, so that opening it does not relieve the digester (the operator is then on the roof when the seal fails)'),
    't_open_vent_min': dict(v=3.0, src='Assumption: minutes from arrival on the roof to an open manual vent; arrivals within the response bucket R(<= 0.5 h) are taken uniform between 5 and 30 min, so the share arriving too late is t_open / 25'),
    'p_seal_dist':   dict(v={8.0: 0.25, 10.0: 0.5, 12.0: 0.25}, src='Assumption: pressure (kPag) at which the roof seal / manhole lets go, 1.6-2.4 x the 5.0 kPag design pressure. The window from the PAH (4.8) to failure is computed hour by hour from the rise rate: isolated digester k_iso q(h)/2, shared header k_full q_net(h)'),
    'k_iso':         dict(v=0.266, src='Pressure model (Appendix IV): stiffness of one isolated digester headspace, kPa per reference m3'),
    'k_full':        dict(v=0.0803, src='Pressure model (Appendix IV): stiffness of the gas side with the inner membrane against the outer one, kPa per reference m3'),
    'q_net_min':     dict(v=5.0, src='Lower bound of the net inflow (m3/h) when the holder is full and the consumer still takes 60 m3/h'),
    't_resp_min_min': dict(v=5.0, src='Assumption: earliest arrival on the roof after an alarm (min); the response CDF rises linearly from here to the BN buckets R(<=0.5 h), R(<=2 h), R(<=8 h)'),
    'tau_seal_s':    dict(v=30.0, src='Assumption (as in consequence.py): duration over which the compressed headspace escapes through the failed seal; the H2S dose of a person in that cloud is taken at 3 m for this time'),
    't_vent_manual_min': dict(v=15.0, src='Assumption: minutes the operator stays beside the manual vent after opening it (41 m3/h, continuous) until the feed stop brings the pressure down'),
    'yH2S_ppm':      dict(v=2527.0, src='PyADM1ODE annual mean of the raw gas (930-5,845 ppm); used when yH2S_mode = mean'),
    'yH2S_mode':     dict(v='hourly', src="'hourly': the PyADM1ODE hourly H2S of the raw gas (930-5,845 ppm) at the hour of the release; 'mean': the annual mean 2,527 ppm. With the steep RIVM probit (n = 6.52) the timing of the H2S peaks matters"),
    'yH2S_scale':    dict(v=1.0, src='Multiplier on the H2S content; 0.2 represents in-situ desulphurisation in the digester headspace (micro-aeration, about 500 ppm on average)'),
    'comp_mode':     dict(v='hourly', src="'hourly': CH4, CO2 and H2O of the raw gas from PyADM1ODE hour by hour (buoyancy, LFL, CO2 dose); 'mean': the annual mean (CH4 64.0 %, CO2 30.2 %, H2O 5.8 %)"),
    'co2_tox':       dict(v=True, src='CO2 toxicity included: HSE dangerous toxic loads (n = 8) SLOT 1.5e40 / SLOD 1.5e41 ppm^8 min as a probit (SLOT = 1 %, SLOD = 50 %); combined with H2S as independent action, P = 1 - (1 - P_H2S)(1 - P_CO2)'),
    'h2s_probit':    dict(v='rivm2018', src="RIVM (2018-01-17, interim) Pr = -7.87 + 0.31 ln(C^6.52 t), C mg/m3 (1 ppm = 1.417 mg/m3), t min. 'bevi2009' = Pr = -11.5 + ln(C^1.9 t) (C mg/m3), the former Dutch value, kept for comparison"),
    'conc_factor':   dict(v=1.0, src='Near-field concentration multiplier: 1 = plume-axis value Q/(pi u sy sz) (reflected, conservative), 0.5 = free plume, 2 = recirculation behind the weather hood or the roof edge'),
    'beta_entrain':  dict(v=0.6, src='Entrainment coefficient of the bent-over plume spread (Briggs 0.6); sensitivity 0.4-0.8'),
    'duty':          dict(v={'R1 chatter (valve open)': 44.0 / 141.0, 'R2 downstream stop': 84.5 / 141.0, 'R3 isolated digester': 41.2 / 141.0, 'R5 blocked PVRV, continuous': 1.0},
                          src='1-s model: every release is realised as pulses at the minimum stable opening (141 m3/h); duty = time-mean rate / 141. The H2S dose uses duty x stay at the open-valve concentration; with the RIVM exponent n = 6.52 this is delta^(1-n) = 640 times the toxic load of a time-mean plume for R1'),
    'generic_f_pvrv': dict(v=0.01, src='Generic-QRA comparison: spurious / demand-driven PVRV opening 1e-2 per valve-year (CCPS-type initiating-event value), 2 valves'),
    'generic_dur_h': dict(v=1.0, src='Generic-QRA comparison: assumed release duration'),
}
CASES = [(1, 1.0), (1, 0.8), (3, 1.0), (3, 0.8)]
BR = ['demand', 'stop', 'foam', 'block', 'wv']
REL = {'demand': 'R1 chatter (valve open)', 'stop': 'R2 downstream stop', 'foam': 'R3 isolated digester', 'block': 'R3 isolated digester', 'wv': 'R1 chatter (valve open)'}
Q_REL = {'R1 chatter (valve open)': 141.0, 'R2 downstream stop': 141.0, 'R3 isolated digester': 141.0, 'R5 blocked PVRV, continuous': 41.2}
WX_MIX = {'D5': ('D', 5.0), 'D3': ('D', 3.0), 'F2': ('F', 2.0), 'calm': (None, 0.5)}
U_EDGES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.5, 1e9]; U_REP = [0.75, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 11.0]   # vent-height wind bins (m/s)
WX_HOURLY = {f'h:{c}{u:g}': (c, u) for c in 'ABCDEF' for u in U_REP}; WX_HOURLY['calm'] = (None, 0.5)
WX = {**WX_MIX, **WX_HOURLY}
STATIC_FACTORS = ('weather', 'daynight', 'q', 'h2s')
_MET = None
def met_hourly():
    global _MET
    if _MET is None:
        import met_stability as ms
        _MET = ms.build()
    return _MET

def wx_keys(P):
    return list(WX_MIX) if P['weather_mode']['v'] == 'mix' else list(WX_HOURLY)

def wx_weights(P, day, dt):
    """weather weights per hour: {key: array(n)}. 'mix' = the assumed day/night mix; 'hourly' = one-hot from the KMA record"""
    if P['weather_mode']['v'] == 'mix':
        return {w: np.where(day, P['weather_day']['v'].get(w, 0.0), P['weather_night']['v'].get(w, 0.0)) for w in WX_MIX}
    m = met_hourly().reindex(pd.DatetimeIndex(dt)).ffill().bfill()
    ub = np.clip(np.digitize(m['u_vent_ms'].to_numpy(), U_EDGES) - 1, 0, len(U_REP) - 1)
    key = np.where(m['calm'].to_numpy() == 1, 'calm', np.array([f'h:{c}{U_REP[i]:g}' for c, i in zip(m['PG_class'], ub)]))
    return {w: (key == w).astype(float) for w in WX_HOURLY if (key == w).any()}

# ----------------------------------------------------------------------------- plume geometry (cached; concentrations scale with the H2S content)
_GEO = {}
COMP_STEP = 0.01                                                                  # composition bins: CH4 mole fraction in steps of 1 %
def comp_props(cb):
    """(reduced gravity, LFL) of the gas in composition bin cb = (yCH4, yCO2, yH2O) rounded; None = annual mean"""
    if cb is None: return None, cq.LFL
    yC, yD, yW = cb; Mg = yC * 0.016043 + yD * 0.044010 + yW * 0.018015 + (1 - yC - yD - yW) * 0.034081
    rho = 101325 * Mg / (8.314462 * 308.15)
    return cq.g_reduced(rho), 0.05 / yC

def geometry(Qh, wname, P, cb=None):
    """(C at 1 m, C at the LSIR distance, C at 3 m: all biogas volume fractions on the plume axis; LFL distance m) for a continuous release Qh m3/h,
    with the buoyancy and LFL of composition bin cb (None: annual mean)"""
    f, be, xd = P['conc_factor']['v'], P['beta_entrain']['v'], P['lsir_distance_m']['v']
    key = (round(Qh, 3), wname, f, be, xd, cb)
    if key not in _GEO:
        cls, u = WX[wname]; Q = Qh / 3600; g1, lfl = comp_props(cb)
        if cls is None:
            c = lambda z: min(1.0, f * float(cq.mtt_plume(Q, z, g1=g1)[0]))
            z = np.linspace(0.3, 30, 300); cz = np.minimum(1.0, f * cq.mtt_plume(Q, z, g1=g1)[0])
            xl = float(z[np.argmax(cz < lfl)]) if (cz < lfl).any() else 30.0
        else:
            kw = dict(factor=f, beta=be, g1=g1)
            c = lambda x: float(cq.plume_centreline(Q, u, cls, x, **kw)); xl = cq.dist_to(lfl, Q, u, cls, **kw)
        _GEO[key] = (c(1.0), c(xd), c(3.0), xl)
    return _GEO[key]

def geo_hourly(Qh, wname, P, comp):
    """the four geometry values hour by hour (arrays) for the composition bins of the year"""
    vals = np.array([geometry(Qh, wname, P, cb) for cb in comp['bins']])            # (n_bins, 4)
    return [vals[comp['idx'], j] for j in range(4)]

def effects(Q_m3h, wname, yH2S=None, P=None):
    """(C at 1 m vol fraction, H2S ppm at 1 m, LFL distance) for a continuous release Q at the vent in weather class wname"""
    P = P or QRA; yH2S = cq.yH2S if yH2S is None else yH2S
    c1, _, _, xl = geometry(Q_m3h, wname, P)
    return c1, c1 * yH2S * 1e6, xl

def probit(C_ppm, t_min, P):
    return cq.h2s_probit_fatality(C_ppm, np.maximum(t_min, 0.1), P['h2s_probit']['v'])

def tox(c, t_min, y, yco2, P):
    """(fatality H2S + CO2 combined as independent action, H2S alone) for biogas fraction c over t_min minutes"""
    ph = probit(c * y, t_min, P)
    if not P['co2_tox']['v']: return ph, ph
    pc = cq.co2_probit_fatality(c * yco2 * 1e6, np.maximum(t_min, 0.1))
    return 1 - (1 - ph) * (1 - pc), ph

def comp_series(P, n, static=False):
    """hourly composition of the raw gas: yCH4, yCO2 (mole fractions) and the composition bins used for the plume geometry"""
    d = pd.read_csv(bn.ADM1_CSV).iloc[:n]
    C = d['CH4_wet_pct'].to_numpy() / 100; D = d['CO2_wet_pct'].to_numpy() / 100; W = d['H2O_wet_pct'].to_numpy() / 100
    if P['comp_mode']['v'] != 'hourly' or static:
        C, D, W = (np.full(n, a.mean()) for a in (C, D, W))
    key = np.round(C / COMP_STEP) * COMP_STEP
    bins, idx = np.unique(key, return_inverse=True)
    bins_t = [(float(b), float(D[idx == i].mean()), float(W[idx == i].mean())) for i, b in enumerate(bins)]
    if P['comp_mode']['v'] != 'hourly' and not static: bins_t = [None]; idx = np.zeros(n, int)
    return dict(yCH4=C, yCO2=D, bins=bins_t, idx=idx)

def h2s_series(P, n):
    """H2S of the raw gas (ppm) hour by hour"""
    if P['yH2S_mode']['v'] == 'hourly':
        y = pd.read_csv(bn.ADM1_CSV)['H2S_gas_ppmv_eq'].to_numpy()[:n]
    else:
        y = np.full(n, P['yH2S_ppm']['v'])
    return y * P['yH2S_scale']['v']

def resp_cdf(prm, wday, t_h, P):
    """P(operator on the roof within t_h after the alarm): piecewise linear through 0 at t_resp_min and the BN buckets R(<=0.5 h), R(<=2 h), R(<=8 h); day/night weighted by wday"""
    t0 = P['t_resp_min_min']['v'] / 60.0
    def F(R):
        xs = [0.0, t0] + list(R.keys()); ys = [0.0, 0.0] + list(np.cumsum(list(R.values())))
        return np.interp(t_h, xs, ys)
    return wday * F(prm['R_day']['v']) + (1 - wday) * F(prm['R_night']['v'])

def overpressure_chain(rate, rise, vf, wday, prm, P, noauto=None):
    """race between the pressure rise (kPa/h) and the operator who must open a manual vent on the roof.
    rate: initiating events per hour; vf: probability that the manual vent is fouled too; noauto(t_h): probability that nothing else
    (automatic flare retry, end of the downstream stop) has relieved the pressure by t_h.
    Returns expected per hour: seal failures, operators beside an open manual vent, operators on the roof at the failure"""
    tov = P['t_open_vent_min']['v'] / 60.0
    fail = np.zeros_like(rate); vent = np.zeros_like(rate); onroof = np.zeros_like(rate)
    for ps, w in P['p_seal_dist']['v'].items():
        tw = (ps - P['PAH_kPag']['v']) / np.maximum(rise, 1e-6)
        Fa = resp_cdf(prm, wday, tw - tov, P); Fb = resp_cdf(prm, wday, tw, P)
        na = noauto(tw) if noauto is not None else 1.0
        prevent = Fa * (1 - vf)
        fail += w * na * (1 - prevent); vent += w * na * prevent; onroof += w * na * ((Fb - Fa) * (1 - vf) + Fb * vf)
    return rate * fail, rate * vent, rate * onroof

def fatality_tables(P, y_ppm=None):
    """P(fatality | exposure mode, release type, weather) at one H2S content (default: the annual mean) - for the tables"""
    y = (P['yH2S_ppm']['v'] * P['yH2S_scale']['v']) if y_ppm is None else y_ppm
    yco2 = cq._props_all['props']['yCO2']
    tab = {}
    for rel, Qh in Q_REL.items():
        for w in wx_keys(P):
            delta = P['duty']['v'][rel]; c1, cd, c3, xl = geometry(Qh, w, P)
            f_overlap = min(1.0, P['t_at_valve_min']['v'] / P['t_fix_min']['v'])
            flash_resp = P['p_zone']['v'] * (P['p_ign_worker']['v'] + P['p_ign']['v'] * f_overlap) * P['p_fat_flash_small']['v']
            h2s_resp = P['p_in_plume_h2s']['v'] * float(tox(c1, P['t_at_valve_min']['v'] * delta, y, yco2, P)[0])
            lsir_ev = (P['p_zone']['v'] * P['p_ign']['v'] * P['p_fat_flash_small']['v'] if xl >= P['lsir_distance_m']['v'] else 0.0) + P['p_in_plume_h2s']['v'] * float(tox(cd, 30.0 * delta, y, yco2, P)[0])
            flash_hour = P['p_zone']['v'] * (P['p_ign_worker']['v'] + P['p_ign']['v']) * P['p_fat_flash_small']['v']
            h2s_hour = P['p_in_plume_h2s']['v'] * float(tox(c1, 60.0 * delta, y, yco2, P)[0])
            tab[(rel, w)] = dict(C_1m=c1, H2S_1m_ppm=c1 * y, CO2_1m_pct=c1 * yco2 * 100, O2_1m_pct=20.95 * (1 - c1), LFL_m=xl, duty=delta, p_fat_responder=flash_resp + h2s_resp, p_fat_flash=flash_resp, p_fat_h2s=h2s_resp,
                                 p_fat_co2_alone=float(cq.co2_probit_fatality(c1 * yco2 * 1e6, P['t_at_valve_min']['v'] * delta)), p_fat_routine_hour=flash_hour + h2s_hour, H2S_lsir_ppm=cd * y, p_fat_lsir_per_event=lsir_ev)
    return tab

def weather_mix(P, day):
    return P['weather_day']['v'] if day else P['weather_night']['v']

_BN = {}
def bn_year(case, f_on, prm_over=None):
    """BN run for one case (cached: the QRA-side sensitivities reuse it)"""
    key = (case, f_on, json.dumps(prm_over or {}, sort_keys=True, default=str))
    if key not in _BN:
        prm = copy.deepcopy(bn.PARAMS); prm['f_on']['v'] = f_on; prm['t_retry_h']['v'] = 0.5; prm['n_retry']['v'] = 4; prm['pressure_backup']['v'] = False
        tag = f'onoff_c{case}_f{int(round(f_on * 100))}'; prm['fill_series']['v'] = f'fill_hourly_{tag}.csv'; prm['demand_series']['v'] = f'flare_demands_{tag}.csv'   # v1.3: 1-s model series per Case
        for k, v in (prm_over or {}).items(): prm[k]['v'] = v
        inp = bn.load_inputs(prm=prm)
        _BN[key] = (prm, inp, bn.run_year(case, inp, prm, occupancy=True))
        if len(_BN) > 40: _BN.pop(next(iter(_BN)))
    return _BN[key]

def run_case(case, f_on, P=None, prm_over=None, static=()):
    """hour-by-hour risk. static: factors from STATIC_FACTORS replaced by their annual mean (static QRA on the same scenarios)"""
    P = P or QRA
    prm, inp, df = bn_year(case, f_on, prm_over)
    n = len(df); day = inp['day'][:n]
    out = df[['h', 'q', 'fill', 'p_open', 'p_state_open', 'v_open', 'ch4_kg']].copy(); out['day'] = day
    out['datetime'] = inp['t'].iloc[:n].to_numpy()
    W = wx_weights(P, day, out['datetime'].to_numpy())
    wday = day.astype(float); q = df['q'].to_numpy().astype(float); y = h2s_series(P, n)
    if 'weather' in static: W = {w: np.full(n, v.mean()) for w, v in W.items()}
    if 'daynight' in static: wday = np.full(n, wday.mean())
    if 'q' in static: q = np.full(n, q.mean())
    if 'h2s' in static: y = np.full(n, y.mean())
    comp = comp_series(P, n, static='h2s' in static); yco2 = comp['yCO2']
    out['yH2S_ppm'] = y; out['yCO2'] = yco2; out['yCH4'] = comp['yCH4']
    pz, pi, piw, pfs, pip = (P[k]['v'] for k in ('p_zone', 'p_ign', 'p_ign_worker', 'p_fat_flash_small', 'p_in_plume_h2s'))
    E = {e: np.zeros(n) for e in ('h2s_resp', 'flash_resp', 'mech_resp', 'h2s_rout', 'flash_rout', 'h2so_resp', 'h2so_rout')}   # 'h2s' = toxic (H2S + CO2), 'h2so' = H2S alone
    tot_lsir = np.zeros(n)
    # ---- fatality per exposure, hour by hour (weather and H2S of that hour)
    fat = {}
    for rel in ('R1 chatter (valve open)', 'R2 downstream stop', 'R3 isolated digester'):
        delta = P['duty']['v'][rel]; Qh = Q_REL[rel]
        f_overlap = min(1.0, P['t_at_valve_min']['v'] / P['t_fix_min']['v'])
        fr_flash = np.zeros(n); fr_h2s = np.zeros(n); fh_flash = np.zeros(n); fh_h2s = np.zeros(n); fl = np.zeros(n); fr_o = np.zeros(n); fh_o = np.zeros(n)
        for w, wd in W.items():
            c1, cd, c3, xl = geo_hourly(Qh, w, P, comp)
            fr_flash += wd * pz * (piw + pi * f_overlap) * pfs
            a, b = tox(c1, P['t_at_valve_min']['v'] * delta, y, yco2, P); fr_h2s += wd * pip * a; fr_o += wd * pip * b
            fh_flash += wd * pz * (piw + pi) * pfs
            a, b = tox(c1, 60.0 * delta, y, yco2, P); fh_h2s += wd * pip * a; fh_o += wd * pip * b
            fl += wd * (np.where(xl >= P['lsir_distance_m']['v'], pz * pi * pfs, 0.0) + pip * tox(cd, 30.0 * delta, y, yco2, P)[0])
        fat[rel] = (fr_flash, fr_h2s, fh_flash, fh_h2s, fl, fr_o, fh_o)
    rout = wday * P['routine_h_per_week']['v'] / (12.0 * 7.0)                # share of the hour spent on the roof (day shift only)
    tot_resp = np.zeros(n); tot_rout = np.zeros(n)
    for k in BR:
        fr_flash, fr_h2s, fh_flash, fh_h2s, fl, fr_o, fh_o = fat[REL[k]]
        p_roof = wday * P['p_roof']['v'][k][0] + (1 - wday) * P['p_roof']['v'][k][1]
        o = df[f'o_{k}'].to_numpy() * p_roof; s_ = df[f's_{k}'].to_numpy() * rout * P['p_near_valve_routine']['v']
        E['flash_resp'] += o * fr_flash; E['h2s_resp'] += o * fr_h2s; E['flash_rout'] += s_ * fh_flash; E['h2s_rout'] += s_ * fh_h2s
        E['h2so_resp'] += o * fr_o; E['h2so_rout'] += s_ * fh_o
        out[f'IR_resp_{k}'] = o * (fr_flash + fr_h2s); out[f'IR_rout_{k}'] = s_ * (fh_flash + fh_h2s); out[f'LSIR_{k}'] = df[f'p_{k}'].to_numpy() * fl
        tot_resp += out[f'IR_resp_{k}'].to_numpy(); tot_rout += out[f'IR_rout_{k}'].to_numpy(); tot_lsir += out[f'LSIR_{k}'].to_numpy()
    # ---- a PVRV that does not relieve: exposure beside the manual vent (prevented) or on the roof at the seal failure
    V0 = cq._props_all['r5']['puff_m3']; Q_seal = V0 / P['tau_seal_s']['v'] * 3600.0
    def vent_fat(Qv):
        fl_ = np.zeros(n); fh_ = np.zeros(n); fo_ = np.zeros(n)
        for w, wd in W.items():
            c1, _, _, _ = geo_hourly(Qv, w, P, comp)
            a, b = tox(c1, P['t_vent_manual_min']['v'], y, yco2, P)
            fl_ += wd * pz * (piw + pi) * pfs; fh_ += wd * pip * a; fo_ += wd * pip * b
        return fl_, fh_, fo_
    seal_h2s = np.zeros(n); seal_o = np.zeros(n)
    for w, wd in W.items():
        _, _, c3, _ = geo_hourly(Q_seal, w, P, comp)
        a, b = tox(c3, P['tau_seal_s']['v'] / 60.0, y, yco2, P)
        seal_h2s += wd * P['p_zone_R5']['v'] * a; seal_o += wd * P['p_zone_R5']['v'] * b     # inside the 30-s depressurisation cloud, 3 m from the failure
    seal_flash = P['p_zone_R5']['v'] * (pi + piw) * P['p_fat_flash_R5']['v']; seal_mech = P['p_injury_seal']['v']
    pfd = bn.pv_pfd(prm); dirty = P['pv_dirty_factor']['v']
    pfd_single = min(1.0, pfd['single'] * dirty); pfd_shared = min(1.0, prm['beta_pv']['v'] * pfd_single + ((1 - prm['beta_pv']['v']) * pfd_single) ** 2)
    if P['second_relief']['v']: pfd_single = pfd_shared                                       # a second device on its own nozzle of each digester
    pfd_fl, pfd_rt, pfd_nh, pD = prm['pfd_flare']['v'], prm['t_retry_h']['v'], int(prm['n_retry']['v']), prm['p_hard']['v']
    def no_retry(t):                                                                            # no successful automatic flare retry by t
        m = np.minimum(pfd_nh, np.floor(np.maximum(t, 0) / pfd_rt)) if pfd_rt > 0 else 0
        return pD + (1 - pD) * pfd_fl ** m
    def stop_lasts(t):
        return np.sum([p * (D > t) for D, p in prm['D_dist']['v'].items()], axis=0)
    p_foam_init = prm['lam_foam']['v'] * prm['p_block_foam']['v'] / 8760
    vf = P['p_vent_fouled']['v']; q_net = np.maximum(q - prm['Q_use']['v'], P['q_net_min']['v'])
    rise_iso = P['k_iso']['v'] * q / 2; rise_net = P['k_full']['v'] * q_net; rise_all = P['k_full']['v'] * q
    chains = {   # name: (initiating events per hour, rise kPa/h, P(manual vent fouled), noauto, release rate at the manual vent m3/h)
        'R5':      (np.full(n, p_foam_init * prm['p_pv_fouled']['v'] * P['p_fouled_factor']['v']), rise_iso, vf, None, 41.2),
        'HW_foam': (df['p_foam'].to_numpy() * pfd_single, rise_iso, vf, None, 41.2),
        'HW_block': (df['p_block'].to_numpy() * pfd_single, rise_iso, 0.0, None, 41.2),
        'HW_demand': (df['p_demand'].to_numpy() * pfd_shared, rise_net, 0.0, no_retry, 41.2),
        'HW_stop': (df['p_stop'].to_numpy() * pfd_shared, rise_all, 0.0, lambda t: no_retry(t) * stop_lasts(t), 84.5),
        'HW_wv':   (df['p_wv'].to_numpy() * pfd_shared, rise_net, 0.0, None, 41.2)}
    vf_cache = {}
    for name, (rate, rise, vfo, na, Qv) in chains.items():
        if Qv not in vf_cache: vf_cache[Qv] = vent_fat(Qv)
        v_flash, v_h2s, v_o = vf_cache[Qv]
        p_fail, p_vent, p_on = overpressure_chain(rate, rise, vfo, wday, prm, P, na)
        out[f'n_init_{name}'] = rate; out[f'p_fail_{name}'] = p_fail
        E['flash_resp'] += p_vent * v_flash + p_on * seal_flash; E['h2s_resp'] += p_vent * v_h2s + p_on * seal_h2s; E['mech_resp'] += p_on * seal_mech
        E['h2so_resp'] += p_vent * v_o + p_on * seal_o
        out[f'IR_resp_{name}'] = p_vent * (v_flash + v_h2s) + p_on * (seal_flash + seal_h2s + seal_mech)
        tot_resp += out[f'IR_resp_{name}'].to_numpy()
        lsir_ = p_fail * (P['p_zone_R5']['v'] * pi * P['p_fat_flash_R5']['v'] + seal_h2s); out[f'LSIR_{name}'] = lsir_; tot_lsir += lsir_
    hw = [c for c in chains if c.startswith('HW_')]
    out['p_R5'] = out['p_fail_R5']; out['p_HW'] = out[[f'p_fail_{c}' for c in hw]].sum(axis=1); out['IR_resp_HW'] = out[[f'IR_resp_{c}' for c in hw]].sum(axis=1)
    out['n_init_HW'] = out[[f'n_init_{c}' for c in hw]].sum(axis=1)
    for e, v in E.items(): out[f'E_{e}'] = v
    out['IR_resp'] = tot_resp; out['IR_rout'] = tot_rout; out['IR_total'] = tot_resp + tot_rout; out['LSIR'] = tot_lsir       # IR_total = PLL over everyone who responds or works on the roof
    out['IR_worker'] = tot_resp / P['n_responders']['v'] + tot_rout                                                        # the most exposed individual: 1/3 of the alarms + all the routine roof work
    return out, fatality_tables(P)

def summarize(out, case, f_on):
    s = dict(case=case, f_on=f_on, openings=out['p_open'].sum(), R5_init=out['n_init_R5'].sum(), R5_events=out['p_R5'].sum(), HW_init=out['n_init_HW'].sum(), HW_events=out['p_HW'].sum(),
             open_h=out['p_state_open'].sum(), vent_m3=out['v_open'].sum(), ch4_kg=out['ch4_kg'].sum(),
             IR_resp=out['IR_resp'].sum(), IR_rout=out['IR_rout'].sum(), IR_total=out['IR_total'].sum(), IR_worker=out['IR_worker'].sum(), LSIR=out['LSIR'].sum())
    for k in BR + ['R5', 'HW']:
        s[f'IR_resp_{k}'] = out[f'IR_resp_{k}'].sum()
    s['IR_day'] = out.loc[out.day, 'IR_total'].sum(); s['IR_night'] = out.loc[~out.day, 'IR_total'].sum()
    n_resp = QRA['n_responders']['v']
    for e in ('h2s', 'flash', 'mech', 'h2so'):
        r_ = out[f'E_{e}_resp'].sum(); o_ = out[f'E_{e}_rout'].sum() if f'E_{e}_rout' in out else 0.0
        s[f'PLL_{e}'] = r_ + o_; s[f'IRw_{e}'] = r_ / n_resp + o_
    s['PLL_co2_increment'] = s['PLL_h2s'] - s['PLL_h2so']; s['IRw_co2_increment'] = s['IRw_h2s'] - s['IRw_h2so']   # 'h2s' columns = toxic (H2S + CO2)
    return s

def generic_qra(P=None):
    """conventional QRA: generic PVRV opening frequency, no time structure, same exposure and consequence assumptions (annual-mean weather and H2S)"""
    P = P or QRA
    f = P['generic_f_pvrv']['v'] * 2
    t = pd.date_range('2020-11-01 01:00', periods=8760, freq='h'); day = (t.hour >= 8) & (t.hour < 20)
    W = {w: float(np.mean(v)) for w, v in wx_weights(P, day, t).items()}       # annual weather frequencies, no time structure
    y = float(h2s_series(P, 8760).mean())
    ft = fatality_tables(P, y)
    pf = sum(W[w] * ft[('R2 downstream stop', w)]['p_fat_responder'] for w in W)
    p_roof = 0.5 * P['p_roof']['v']['stop'][0] + 0.5 * P['p_roof']['v']['stop'][1]
    lsir = sum(W[w] * ft[('R2 downstream stop', w)]['p_fat_lsir_per_event'] for w in W)
    return dict(openings=f, IR_resp=f * p_roof * pf, IR_total=f * p_roof * pf, IR_worker=f * p_roof * pf / P['n_responders']['v'], LSIR=f * lsir, vent_m3=f * 84.5 * P['generic_dur_h']['v'])

def with_direct_indication(P=None):
    """design option A: PVRV position switch or a fixed CH4/H2S detector at the vent. The operator knows the valve is venting,
    so nobody climbs to a venting valve without protection: roof presence while venting x0.2, prolonged stay in the plume x0.25"""
    P2 = copy.deepcopy(P or QRA)
    P2['p_roof']['v'] = {k: (v[0] * 0.2, v[1] * 0.2) for k, v in P2['p_roof']['v'].items()}
    P2['p_in_plume_h2s']['v'] = P2['p_in_plume_h2s']['v'] * 0.25
    return P2

def with_second_relief(P=None):
    """design option B: an independent second relief path (second PVRV or rupture disc on its own nozzle, or a remotely operated vent valve).
    Common-cause fouling of both devices by the same foam: beta = 0.1; hardware fail-to-open of an isolated digester becomes 1oo2;
    nobody needs to open a manual vent on the roof"""
    P2 = copy.deepcopy(P or QRA)
    P2['p_fouled_factor']['v'] = P2['p_fouled_factor']['v'] * 0.1
    P2['p_vent_fouled']['v'] = 0.0; P2['second_relief']['v'] = True
    return P2

def warning_times(P=None):
    """minutes between the PAH alarm and the PVRV opening, by case and net inflow (holder full: k_full; one digester blocked: k_iso)"""
    P = P or QRA; rows = []
    for case, p_op in ((1, 4.0), (2, 3.0), (3, 2.0)):
        for q_net, k, what in ((25.0, P['k_full']['v'], 'holder full, surplus 25 m3/h (production 85)'), (55.0, P['k_full']['v'], 'holder full, surplus 55 m3/h (production 115)'), (41.2, P['k_iso']['v'], 'one digester blocked, 41 m3/h')):
            t_alarm = (P['PAH_kPag']['v'] - p_op) / (k * q_net) * 60; t_open = (5.05 - p_op - 0.11) / (k * q_net) * 60
            t_seal = {ps: (ps - P['PAH_kPag']['v']) / (k * q_net) * 60 for ps in P['p_seal_dist']['v']}
            rows.append(dict(case=case, p_op=p_op, situation=what, t_alarm_min=t_alarm, t_open_min=t_open, warning_min=t_open - t_alarm,
                             **{f'PAH_to_seal_{ps:g}kPag_min': v for ps, v in t_seal.items()}))
    return pd.DataFrame(rows)

def static_vs_dynamic(cases=CASES, P=None):
    """same scenarios, same annual event numbers: static QRA (all time-dependent factors at their annual mean), dynamic, and the
    dynamic result with one factor averaged at a time (its share of the time-correlation effect)"""
    P = P or QRA; rows = []
    for case, f_on in cases:
        dyn = summarize(run_case(case, f_on, P)[0], case, f_on); sta = summarize(run_case(case, f_on, P, static=STATIC_FACTORS)[0], case, f_on)
        r = dict(case=case, f_on=f_on, dynamic=dyn['IR_total'], static=sta['IR_total'], ratio=dyn['IR_total'] / sta['IR_total'],
                 dynamic_worker=dyn['IR_worker'], static_worker=sta['IR_worker'], dynamic_h2s=dyn['PLL_h2s'], static_h2s=sta['PLL_h2s'])
        for f in STATIC_FACTORS:
            r[f'dynamic_without_{f}'] = summarize(run_case(case, f_on, P, static=(f,))[0], case, f_on)['IR_total']
            r[f'static_with_{f}'] = summarize(run_case(case, f_on, P, static=tuple(x for x in STATIC_FACTORS if x != f))[0], case, f_on)['IR_total']
        rows.append(r)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    rows = []; profiles = {}
    wt = warning_times(); wt.to_csv(os.path.join(OUT, 'alarm_warning_times.csv'), index=False, float_format='%.1f'); print(wt.round(1).to_string(index=False))
    for case, f_on in CASES:
        out, ft = run_case(case, f_on)
        out.to_csv(os.path.join(OUT, f'risk_hourly_case{case}_f{int(f_on * 100)}.csv.gz'), index=False, float_format='%.6g')
        rows.append(summarize(out, case, f_on)); profiles[(case, f_on)] = out
        s = rows[-1]; print(f'Case {case}, flare start {int(f_on * 100)} %: openings {s["openings"]:.2f}, seal failures R5 {s["R5_events"]:.2e} HW {s["HW_events"]:.2e}, PLL {s["IR_total"]:.2e} (H2S {s["PLL_h2s"]:.2e}, flash {s["PLL_flash"]:.2e}, mech {s["PLL_mech"]:.2e}), IR_worker {s["IR_worker"]:.2e}', flush=True)
    for name, fn in (('direct indication', with_direct_indication), ('second relief', with_second_relief), ('both', lambda P=None: with_second_relief(with_direct_indication(P)))):
        for case, f_on in CASES:                                   # design options: A) direct indication of a venting PVRV, B) independent second relief path, A+B
            s_ = summarize(run_case(case, f_on, fn())[0], case, f_on); s_['case'] = f'{case} + {name}'; rows.append(s_)
    r = pd.DataFrame(rows); g = generic_qra(); g.update(case='generic', f_on=np.nan); r = pd.concat([r, pd.DataFrame([g])], ignore_index=True)
    r.to_csv(os.path.join(OUT, 'risk_summary.csv'), index=False, float_format='%.4g')
    pd.set_option('display.width', 250); print(r[['case', 'f_on', 'openings', 'R5_events', 'HW_events', 'IR_total', 'IR_worker', 'PLL_h2s', 'PLL_flash', 'PLL_mech', 'LSIR']].to_string(index=False))
    pd.DataFrame([dict(release=k[0], weather=k[1], **v) for k, v in fatality_tables(QRA).items()]).to_csv(os.path.join(OUT, 'fatality_table.csv'), index=False, float_format='%.4g')
    # effect breakdown (PLL and the most exposed individual) and without a personal H2S detector
    eb = []
    for case, f_on in CASES:
        s = [x for x in rows if x['case'] == case and x['f_on'] == f_on][0]
        P3 = copy.deepcopy(QRA); P3['p_in_plume_h2s']['v'] = 0.3; s3 = summarize(run_case(case, f_on, P3)[0], case, f_on)
        eb.append(dict(case=case, f_on=f_on, PLL=s['IR_total'], PLL_h2s=s['PLL_h2s'], PLL_flash=s['PLL_flash'], PLL_mech=s['PLL_mech'], IRw=s['IR_worker'], IRw_h2s=s['IRw_h2s'], IRw_flash=s['IRw_flash'], IRw_mech=s['IRw_mech'],
                       PLL_no_detector=s3['IR_total'], PLL_h2s_no_detector=s3['PLL_h2s'], IRw_no_detector=s3['IR_worker']))
    pd.DataFrame(eb).to_csv(os.path.join(OUT, 'risk_by_effect.csv'), index=False, float_format='%.4g'); print(pd.DataFrame(eb).to_string(index=False))
    # monthly / hour-of-day profiles
    prof = []
    for (case, f_on), out in profiles.items():
        d = out.set_index('datetime')
        m = d[['IR_total', 'LSIR', 'p_open', 'ch4_kg', 'E_h2s_resp']].resample('MS').sum(); m['yH2S_ppm'] = d['yH2S_ppm'].resample('MS').mean(); m['case'] = case; m['f_on'] = f_on; m['kind'] = 'month'; prof.append(m.reset_index())
        hd = d[['IR_total', 'LSIR', 'p_open', 'p_state_open']].groupby(d.index.hour).sum(); hd['case'] = case; hd['f_on'] = f_on; hd['kind'] = 'hour'; prof.append(hd.reset_index().rename(columns={'datetime': 'hour'}))
    pd.concat(prof, ignore_index=True).to_csv(os.path.join(OUT, 'risk_profiles.csv'), index=False, float_format='%.6g')
    json.dump({k: v for k, v in QRA.items()}, open(os.path.join(OUT, 'qra_params.json'), 'w'), indent=1, default=str)
    # static vs dynamic on the same scenarios
    sd = static_vs_dynamic(); sd.to_csv(os.path.join(OUT, 'static_vs_dynamic.csv'), index=False, float_format='%.4g'); print(sd.round(8).to_string(index=False))
    # tornado on the QRA assumptions, Case 1 x (100 %, 80 %)
    N = cq.PROBIT_H2S[QRA['h2s_probit']['v']][2]
    SENS = [('ignition probability 0.005 / 0.05', {'p_ign': 0.005}, {'p_ign': 0.05}), ('worker ignition source 0 / 0.01', {'p_ign_worker': 0.0}, {'p_ign_worker': 0.01}),
            ('fatality given flash in the small plume 0.03 / 1.0', {'p_fat_flash_small': 0.03}, {'p_fat_flash_small': 1.0}),
            ('P(inside flammable plume at the valve) 0.03 / 0.3', {'p_zone': 0.03}, {'p_zone': 0.3}), ('responder goes to the roof: all branches x0.5 / x2 (capped at 1)', {'p_roof': {k: (v[0] * 0.5, v[1] * 0.5) for k, v in QRA['p_roof']['v'].items()}}, {'p_roof': {k: (min(1, v[0] * 2), min(1, v[1] * 2)) for k, v in QRA['p_roof']['v'].items()}}),
            ('time at the valve 2 / 15 min', {'t_at_valve_min': 2.0}, {'t_at_valve_min': 15.0}), ('prolonged stay in the H2S plume 0.005 / 0.3 (no detector)', {'p_in_plume_h2s': 0.005}, {'p_in_plume_h2s': 0.3}),
            ('H2S probit: Bevi 2009 / RIVM 2018 (base)', {'h2s_probit': 'bevi2009'}, {}),
            ('H2S content: in-situ desulphurisation x0.2 / x1.5', {'yH2S_scale': 0.2}, {'yH2S_scale': 1.5}),
            ('H2S: annual mean 2,527 ppm / hourly ADM1 (base)', {'yH2S_mode': 'mean'}, {}),
            ('CO2 toxicity: excluded / included (base)', {'co2_tox': False}, {}),
            ('CH4/CO2/H2O: annual mean / hourly ADM1 (base)', {'comp_mode': 'mean'}, {}),
            ('duty cycle ignored (time-mean plume) / pulses at full stay', {'duty': {k: (v ** N if v < 1 else 1.0) for k, v in QRA['duty']['v'].items()}}, {'duty': {k: 1.0 for k in QRA['duty']['v']}}),
            ('near-field concentration: free plume x0.5 / recirculation x2', {'conc_factor': 0.5}, {'conc_factor': 2.0}),
            ('plume entrainment beta 0.8 / 0.4', {'beta_entrain': 0.8}, {'beta_entrain': 0.4}),
            ('weather: KMA hourly (base) / former assumed day-night mix', {}, {'weather_mode': 'mix'}),
            ('weather: all D5 / all calm', {'weather_mode': 'mix', 'weather_day': {'D5': 1.0}, 'weather_night': {'D5': 1.0}}, {'weather_mode': 'mix', 'weather_day': {'calm': 1.0}, 'weather_night': {'calm': 1.0}}),
            ('routine roof work 0.5 / 6 h per week', {'routine_h_per_week': 0.5}, {'routine_h_per_week': 6.0}), ('seal-failure injury 0.02 / 0.3', {'p_injury_seal': 0.02}, {'p_injury_seal': 0.3}),
            ('seal fails at 12 / 8 kPag', {'p_seal_dist': {12.0: 1.0}}, {'p_seal_dist': {8.0: 1.0}}),
            ('manual vent fouled too 0.1 / 0.6', {'p_vent_fouled': 0.1}, {'p_vent_fouled': 0.6}), ('near the valve during routine work 0.1 / 1.0', {'p_near_valve_routine': 0.1}, {'p_near_valve_routine': 1.0}),
            ('PVRV fail-to-open: clean (base) / dirty service x5', {}, {'pv_dirty_factor': 5.0}),
            ('PAH available (base) / no PAH: roof visits x1.5, plume stay x2 (foam, block)', {}, {'p_roof': {**QRA['p_roof']['v'], 'foam': (1.0, 0.75), 'block': (1.0, 0.75)}, 'p_in_plume_h2s': 0.04})]
    BSENS = [('flare PFD 0.01 / 0.10', {'pfd_flare': 0.01}, {'pfd_flare': 0.10}), ('foaming 1 / 3 per year', {'lam_foam': 1.0}, {'lam_foam': 3.0}), ('hard-fault share 0 / 0.3', {'p_hard': 0.0}, {'p_hard': 0.3}),
             ('PVRV fouled by foam 0.03 / 0.3', {'p_pv_fouled': 0.03}, {'p_pv_fouled': 0.3}), ('PVRV lambda_DU 1.1e-6 (base) / 2.2e-6 per h', {}, {'p_pv_stuck': 0.0096}),
             ('common cause beta 0.05 / 0.2', {'beta_pv': 0.05}, {'beta_pv': 0.2})]
    srows = []
    for case, f_on in ((1, 1.0), (1, 0.8)):
        base = summarize(run_case(case, f_on)[0], case, f_on)['IR_total']
        for name, lo, hi in SENS:
            P1 = copy.deepcopy(QRA); P2 = copy.deepcopy(QRA)
            for k, v in lo.items(): P1[k]['v'] = v
            for k, v in hi.items(): P2[k]['v'] = v
            srows.append(dict(case=case, f_on=f_on, side='QRA', parameter=name, base=base, low=summarize(run_case(case, f_on, P1)[0], case, f_on)['IR_total'], high=summarize(run_case(case, f_on, P2)[0], case, f_on)['IR_total']))
        for name, lo, hi in BSENS:
            srows.append(dict(case=case, f_on=f_on, side='BN', parameter=name, base=base, low=summarize(run_case(case, f_on, prm_over=lo)[0], case, f_on)['IR_total'], high=summarize(run_case(case, f_on, prm_over=hi)[0], case, f_on)['IR_total']))
        print(f'tornado case {case} {f_on} done', flush=True)
    sd = pd.DataFrame(srows); sd.to_csv(os.path.join(OUT, 'risk_sensitivity.csv'), index=False, float_format='%.4g'); print(sd.round(8).to_string(index=False))
