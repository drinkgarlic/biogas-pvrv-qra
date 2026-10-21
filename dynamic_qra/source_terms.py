# -*- coding: utf-8 -*-
"""
Dynamic QRA, stage 1: source terms of the PVRV releases, extracted from the 1-s pressure model outputs and the BN physics.

Release types
  R1 chatter      flare ignition failure with the holder full (on/off, flare start 100 %): the valve cycles at its minimum stable
                  opening until the automatic retry succeeds -> short pulsed releases (per-episode statistics from the 1-s runs)
  R2 continuous   downstream stop: all production leaves through the PVRV until the downstream or the flare recovers
                  (rate from the event scan, duration from the BN recovery distribution)
  R3 isolated     one digester outlet blocked (foam / condensate / freezing / XV): that digester's production (q/2) leaves through
                  its own PVRV until the operator responds
  R4 ratchet      vent valve stuck + atmospheric ratchet: small intermittent excess (BN)
  R5 blocked      PVRV fouled or stuck -> no relief -> digester pressure rises at k_iso*q/2 (~11 kPa/h) until a roof seal /
                  manhole fails (assumed at 2 x design pressure) -> compressed headspace inventory as a puff + continuous q/2
Composition from the PyADM1ODE hourly series (CH4 ~64 %, H2S ~2,500 ppm, NH3 ~18 ppm).
"""
import os, sys, json, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
PV = os.environ.get('PVRV_PRESSURE_OUT', os.path.join(ROOT, 'upstream', 'pressure_model', 'out')); UP = os.path.join(ROOT, 'upstream', 'pressure_model_results'); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
HAVE_RUNS = os.path.exists(os.path.join(PV, 'onoff_case1_f100_pfd5_events.csv')) and os.path.exists(os.path.join(PV, 'stop_case1_f85.csv.gz'))
if not HAVE_RUNS:
    print('pressure-model run outputs not found (set PVRV_PRESSURE_OUT to a biogas-pvrv-BN pressure_model/out folder to use them): using the frozen tables in upstream/pressure_model_results')
R_GAS, P_REF, T_REF = 8.314462, 101.325, 293.15
M = dict(CH4=0.016043, CO2=0.044010, H2O=0.018015, H2S=0.034081, NH3=0.017031)

adm = pd.read_csv(os.path.join(ROOT, 'upstream', 'data', 'ADM1_hourly_1to9_Q200_6000m3_ffOTS1.csv'), parse_dates=['datetime_interval_end']).set_index('datetime_interval_end')
q_h = adm['Q_gas_wet_m3_d_20C'] / 24

def composition(ts=None):
    """mole fractions (wet) at a timestamp, or the annual mean"""
    row = adm.iloc[adm.index.get_indexer([ts], method='nearest')[0]] if ts is not None else adm.mean(numeric_only=True)
    y = dict(CH4=row['CH4_wet_pct'] / 100, CO2=row['CO2_wet_pct'] / 100, H2O=row['H2O_wet_pct'] / 100, H2S=row['H2S_gas_ppmv_eq'] * 1e-6, NH3=row['NH3_gas_ppmv_eq'] * 1e-6)
    Mg = sum(y[k] * M[k] for k in y)
    return y, Mg

def density(Mg, T_K, p_kPa=101.325):
    return p_kPa * 1000 * Mg / (R_GAS * T_K)

def lfl_biogas(yCH4):
    """lower flammability limit of the mixture (vol fraction of biogas in air): Le Chatelier with CH4 as the only fuel, CO2/H2O inert"""
    return 0.05 / yCH4           # 5.0 vol % CH4 in air

# ----------------------------------------------------------------------------- R1: chattering episodes from the on/off 100 % runs
rows = []
for case in ((1, 2, 3) if HAVE_RUNS else ()):
    tag = f'onoff_case{case}_f100_pfd5'
    ev = pd.read_csv(os.path.join(PV, f'{tag}_events.csv'), parse_dates=['datetime'])
    d = pd.read_csv(os.path.join(PV, f'{tag}_1min.csv.gz'), parse_dates=['datetime']).set_index('datetime')
    op = ev[ev.code.isin([1, 3])].copy(); gaps = op['t_s'].diff().fillna(1e9); op['epi'] = (gaps > 600).cumsum()
    cl = ev[ev.code.isin([2, 4])]
    for e, g in op.groupby('epi'):
        t0 = g['datetime'].min(); t1 = cl[cl['datetime'] >= g['datetime'].max()]['datetime'].min()
        if pd.isna(t1): t1 = g['datetime'].max()
        w = d.loc[t0.floor('min'):t1.ceil('min')]
        vol = w['Q_PV_m3h'].sum() / 60; open_s = w['PV_open_frac'].sum() * 60
        y, Mg = composition(t0)
        rows.append(dict(case=case, start=t0, end=t1, duration_min=(t1 - t0).total_seconds() / 60, cycles=len(g), open_s=open_s, volume_m3=vol,
                         rate_mean_open_m3h=vol / max(open_s, 1) * 3600, rate_1min_peak_m3h=w['Q_PV_m3h'].max(), q_gen_m3h=w['Q_gen_m3h'].mean(),
                         yCH4=y['CH4'], yH2S_ppm=y['H2S'] * 1e6, CH4_kg=vol * y['CH4'] * density(M['CH4'], T_REF), hour=t0.hour, month=t0.month))
epi = pd.DataFrame(rows) if rows else pd.read_csv(os.path.join(UP, 'episodes_R1_f100.csv'), parse_dates=['start', 'end'])
epi.to_csv(os.path.join(OUT, 'episodes_R1_f100.csv'), index=False, float_format='%.3f')
print('R1 episodes per case:', epi.groupby('case').size().to_dict())
print(epi.groupby('case')[['duration_min', 'cycles', 'volume_m3', 'rate_mean_open_m3h', 'rate_1min_peak_m3h', 'CH4_kg']].describe(percentiles=[.5, .95]).round(1).to_string())

# ----------------------------------------------------------------------------- R2: downstream stop (event scan, Case 1 start fill 85 %)
if HAVE_RUNS:
    sc = pd.concat([pd.read_csv(os.path.join(PV, f'stop_case{c}_f85.csv.gz'), parse_dates=['start']).assign(case=c) for c in (1, 2, 3)])
    r2 = sc.groupby('case').agg(rate_1h_med=('vent_1h_after_open_m3', 'median'), rate_1h_p95=('vent_1h_after_open_m3', lambda s: s.quantile(.95)),
                                rate_1h_max=('vent_1h_after_open_m3', 'max'), opens_1h=('opens_1h_after_open', 'median'), open_s_1h=('open_s_1h_after_open', 'median'))
else:
    ss = pd.read_csv(os.path.join(UP, 'summary_stop.csv')); ss = ss[ss.f0 == 85].set_index('case')
    r2 = pd.DataFrame(dict(rate_1h_med=ss['vent_1h'], rate_1h_p95=ss['vent_1h_max'], rate_1h_max=ss['vent_1h_max'], opens_1h=ss['opens_1h'], open_s_1h=ss['open_s_1h']))
print('R2 (downstream stop) vented m3 in the first hour after opening:\n', r2.round(1).to_string())

# ----------------------------------------------------------------------------- R3 / R5: single-digester production and the blocked-PVRV rise
q_one = q_h / 2
k_iso = 0.266                     # kPa per reference m3 (headspace 400 m3)
p_design, p_fail = 5.0, 10.0      # kPag: PVRV set = design pressure; loss of containment assumed at 2 x design
rise_kPa_h = k_iso * q_one        # pressure rise rate of a blocked digester
t_fail_h = (p_fail - p_design) / rise_kPa_h
V_hs = 400.0
puff_m3 = V_hs * (p_fail / 101.325)      # gas expelled by the depressurisation from p_fail to atmospheric (reference m3)
print(f'R3 single-digester rate q/2: mean {q_one.mean():.1f}, p5 {q_one.quantile(.05):.1f}, p95 {q_one.quantile(.95):.1f} m3/h')
print(f'R5 blocked PVRV: rise {rise_kPa_h.mean():.1f} kPa/h (range {rise_kPa_h.min():.1f}-{rise_kPa_h.max():.1f}), time 5 -> 10 kPa: median {t_fail_h.median()*60:.0f} min, puff {puff_m3:.0f} m3')

# ----------------------------------------------------------------------------- composition and properties
y, Mg = composition()
rho35 = density(Mg, 308.15); rho15_air = density(0.028964, 288.15)
lfl = lfl_biogas(y['CH4'])
props = dict(yCH4=y['CH4'], yCO2=y['CO2'], yH2O=y['H2O'], yH2S_ppm=y['H2S'] * 1e6, yNH3_ppm=y['NH3'] * 1e6, M_gas_kg_kmol=Mg * 1000, rho_gas_35C=rho35, rho_air_15C=rho15_air,
             rel_density=rho35 / rho15_air, LFL_biogas_volfrac=lfl, UFL_biogas_volfrac=0.15 / y['CH4'],
             yCH4_range=(adm['CH4_wet_pct'].min() / 100, adm['CH4_wet_pct'].max() / 100), H2S_ppm_range=(adm['H2S_gas_ppmv_eq'].min(), adm['H2S_gas_ppmv_eq'].max()))
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in props.items()})

# ----------------------------------------------------------------------------- source-term table
def row(rtype, case, mode, rate_m3h, duration_h, volume_m3, note, h_release=18.25, puff=0.0):
    return dict(release=rtype, case=case, mode=mode, rate_m3h=rate_m3h, duration_h=duration_h, volume_m3=volume_m3, puff_m3=puff, height_m=h_release,
                rate_kg_s=rate_m3h / 3600 * density(Mg, T_REF), CH4_kg=volume_m3 * y['CH4'] * density(M['CH4'], T_REF), H2S_g=volume_m3 * y['H2S'] * density(M['H2S'], T_REF) * 1000, note=note)
st = []
for c in (1, 2, 3):
    e = epi[epi.case == c]
    if len(e):
        st.append(row('R1 chatter', c, 'pulsed (valve at 10 % min. opening, ~138 cycles/h)', float(e.rate_1min_peak_m3h.median()), float(e.duration_min.median() / 60), float(e.volume_m3.median()),
                      f'per episode, median of {len(e)}; time-mean rate while open {e.rate_mean_open_m3h.median():.0f} m3/h; P95 volume {e.volume_m3.quantile(.95):.0f} m3'))
    st.append(row('R2 downstream stop', c, 'continuous (production), chattering', float(r2.loc[c, 'rate_1h_med']), np.nan, np.nan, 'rate = production (median of scan); duration = recovery (BN: 0.5-24 h)'))
st.append(row('R3 isolated digester', 0, 'continuous (q/2)', float(q_one.mean()), np.nan, np.nan, f'rate q/2 mean {q_one.mean():.0f} (p5-p95 {q_one.quantile(.05):.0f}-{q_one.quantile(.95):.0f}); duration = operator response 0.5 / 2 / 8 h -> {q_one.mean()*0.5:.0f} / {q_one.mean()*2:.0f} / {q_one.mean()*8:.0f} m3'))
st.append(row('R4 ratchet', 0, 'intermittent', 10.0, np.nan, 4.0, 'BN: 2-5 m3 per event over the remaining latent period'))
st.append(row('R5 blocked PVRV', 0, 'puff + continuous (q/2) at roof seal', float(q_one.mean()), np.nan, np.nan, f'LoC assumed at {p_fail} kPag ({t_fail_h.median()*60:.0f} min after blockage); puff {puff_m3:.0f} m3 then q/2 at roof level', h_release=16.5, puff=puff_m3))
st = pd.DataFrame(st); st.to_csv(os.path.join(OUT, 'source_terms.csv'), index=False, float_format='%.4f')
json.dump(dict(props=props, r5=dict(k_iso=k_iso, p_design=p_design, p_fail=p_fail, rise_kPa_h_mean=float(rise_kPa_h.mean()), t_fail_min_median=float(t_fail_h.median() * 60), puff_m3=float(puff_m3))),
          open(os.path.join(OUT, 'source_props.json'), 'w'), indent=1, default=float)
pd.set_option('display.width', 250); print(st.round(3).to_string(index=False))
