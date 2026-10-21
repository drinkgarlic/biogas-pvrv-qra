# -*- coding: utf-8 -*-
"""
Dynamic QRA, stage 2: consequence models for the PVRV releases (near-field, low-momentum, buoyant biogas).

Models (all analytical, documented so that PHAST / ALOHA can be used as a cross-check)
  1. Bent-over buoyant plume for a continuous release in wind: C(x) = Q / (pi * u * sy(x) * sz(x)). The spread combines the
     ambient Pasquill-Gifford sigmas (Briggs rural) with the plume's own entrainment, b = 0.6 z_rise (Briggs bent-over plume,
     z_rise = 1.6 B^(1/3) x^(2/3) / u) and the source radius, added in quadrature. This matters in the first metres and in
     light wind, where ambient turbulence alone would under-predict the dilution of a buoyant plume (relative density 0.80).
     Concentrations are reported on the plume axis (conservative for a receptor at the vent height).
  2. Morton-Taylor-Turner pure plume for calm conditions: centreline dilution of a buoyant plume rising vertically,
     entrainment coefficient alpha = 0.1 (Gaussian profiles) -> concentration vs height and plume radius.
  3. Depressurisation of a failed roof seal (R5): the compressed headspace (39 m3) is treated as a 30-s transient release with
     model 1 (base case); the TNO instantaneous Gaussian puff is also evaluated as an upper bound (it ignores the puff's own
     buoyant mixing and gives much longer distances).
Effects
  * flash fire: a person inside the LFL contour at ignition -> fatality probability 1 (TNO Green Book, outdoors)
  * H2S: RIVM (2018, interim) probit Pr = -7.87 + 0.31 ln(C^6.52 t), C in mg/m3, t in min (validity of the data 1-60 min).
    The ppm values of the plume model are converted with 34.08 / 24.06 = 1.417 mg/m3 per ppm (20 C, 1 atm).
    The former Bevi (2009) probit Pr = -11.5 + ln(C^1.9 t) (C in mg/m3) is kept as model='bevi2009' for comparison;
    up to v1.0 of this repository it was applied, wrongly, with C in ppm.
Outputs: out/effect_distances.csv (per release type x weather), fig/consequence.png
"""
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'out'); FIG = os.path.join(HERE, 'fig'); os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
_pp = os.path.join(OUT, 'source_props.json')
if not os.path.exists(_pp): _pp = os.path.join(HERE, '..', 'results', 'tables', 'source_props.json')
_props_all = json.load(open(_pp)); props = _props_all['props']
yCH4, yH2S, LFL = props['yCH4'], props['yH2S_ppm'] * 1e-6, props['LFL_biogas_volfrac']
RHO_G, RHO_A = props['rho_gas_35C'], props['rho_air_15C']
G = 9.80665
D_VENT = 0.20                          # DN200 vent outlet
A_VENT = math.pi * D_VENT ** 2 / 4
IDLH_H2S, ERPG2_H2S, ERPG3_H2S = 100.0, 30.0, 100.0   # ppm (AEGL-3 10 min: 76 ppm; IDLH 100 ppm)
AEGL3_10MIN = 76.0

# --------------------------------------------------------------------------- Pasquill-Gifford (Briggs rural) sigmas
_PG = {'A': ((0.22, 0.0001, -0.5), (0.20, 0.0, 0.0)), 'B': ((0.16, 0.0001, -0.5), (0.12, 0.0, 0.0)), 'C': ((0.11, 0.0001, -0.5), (0.08, 0.0002, -0.5)),
       'D': ((0.08, 0.0001, -0.5), (0.06, 0.0015, -0.5)), 'E': ((0.06, 0.0001, -0.5), (0.03, 0.0003, -1.0)), 'F': ((0.04, 0.0001, -0.5), (0.016, 0.0003, -1.0))}
def sig(x, cls):
    (a, b, c), (d, e, f) = _PG[cls]
    return a * x * (1 + b * x) ** c, d * x * (1 + e * x) ** f

G1 = G * (RHO_A - RHO_G) / RHO_A                           # reduced gravity of the buoyant gas (m/s2)
def plume_centreline(Q, u, cls, x, r0=D_VENT / 2, beta=0.6, factor=1.0):
    """volume fraction on the plume axis at downwind distance x (m); Q in m3/s, u in m/s.
    sigma = sqrt(sigma_PG^2 + sigma_plume^2 + (r0/2)^2), sigma_plume = beta * z_rise / sqrt(2), z_rise = 1.6 B^(1/3) x^(2/3) / u (Briggs)."""
    x = np.asarray(x, float)
    B = G1 * Q
    z_rise = 1.6 * B ** (1 / 3) * x ** (2 / 3) / u
    sp = beta * z_rise / math.sqrt(2)
    sy0, sz0 = sig(np.maximum(x, 1e-3), cls)
    sy = np.sqrt(sy0 ** 2 + sp ** 2 + (r0 / 2) ** 2); sz = np.sqrt(sz0 ** 2 + sp ** 2 + (r0 / 2) ** 2)
    return np.minimum(1.0, factor * Q / (math.pi * u * sy * sz))      # factor 0.5 = free plume 1/(2 pi); 1 = the base (axis value doubled, roof reflection); 2 = recirculation behind the hood / roof edge

def plume_sigma(Q, u, cls, x, r0=D_VENT / 2, beta=0.6):
    B = G1 * Q; z_rise = 1.6 * B ** (1 / 3) * x ** (2 / 3) / u; sp = beta * z_rise / math.sqrt(2)
    sy0, sz0 = sig(max(x, 1e-3), cls)
    return math.sqrt(sy0 ** 2 + sp ** 2 + (r0 / 2) ** 2), math.sqrt(sz0 ** 2 + sp ** 2 + (r0 / 2) ** 2), z_rise

def dist_to(Cstar, Q, u, cls, xmax=200.0, **kw):
    """downwind distance where the centreline concentration falls to Cstar (vol fraction); 0 if already below at the vent"""
    x = np.logspace(-2, math.log10(xmax), 2000); c = plume_centreline(Q, u, cls, x, **kw)
    if c[0] < Cstar: return 0.0
    i = np.argmax(c < Cstar)
    return float(x[i]) if i > 0 else xmax

def mtt_plume(Q, z, alpha=0.1):
    """calm air, pure buoyant plume (Morton-Taylor-Turner, Gaussian profiles): centreline volume fraction and 1/e radius at height z above the vent"""
    g1 = G * (RHO_A - RHO_G) / RHO_A                      # reduced gravity (m/s2), gas lighter than air
    B = g1 * Q                                            # buoyancy flux (m4/s3)
    z = np.maximum(np.asarray(z, float), 0.3)
    b = 6 * alpha / 5 * z                                 # plume radius (top-hat), m
    w = (5 / (6 * alpha)) * (9 * alpha * B / 10) ** (1 / 3) * z ** (-1 / 3)   # top-hat velocity
    Qz = math.pi * b ** 2 * w                             # volume flux at height z
    return np.minimum(1.0, Q / Qz), b

def puff_peak(V0, u, cls, t, r0):
    """instantaneous puff: peak (centre) volume fraction at time t; sigmas = PG puff values with the initial radius added in quadrature"""
    x = u * t
    sy, sz = sig(np.maximum(x, 0.1), cls)
    sy = np.sqrt((0.5 * sy) ** 2 + (r0 / 2) ** 2); sz = np.sqrt((0.5 * sz) ** 2 + (r0 / 2) ** 2); sx = sy
    return np.minimum(1.0, V0 / ((2 * math.pi) ** 1.5 * sx * sy * sz)), x

MG_PER_PPM_H2S = 34.08 / 24.06                         # mg/m3 per ppm at 20 C
PROBIT_H2S = {'rivm2018': (-7.87, 0.31, 6.52),            # RIVM 2018-01-17 interim, C mg/m3, t min
              'bevi2009': (-11.5, 1.0, 1.9)}              # Reference Manual Bevi 3.2 (2009), C mg/m3, t min
PROBIT_MODEL = 'rivm2018'
def h2s_probit_fatality(C_ppm, t_min, model=None):
    """fatality probability Phi(Pr - 5) for H2S at C_ppm (converted to mg/m3) over t_min minutes"""
    a, b, n = PROBIT_H2S[model or PROBIT_MODEL]
    C = np.maximum(np.asarray(C_ppm, float) * MG_PER_PPM_H2S, 1e-9); t = np.maximum(np.asarray(t_min, float), 1e-6)
    Pr = a + b * (n * np.log(C) + np.log(t))
    return norm.cdf(Pr - 5.0)

def h2s_lc(P, t_min, model=None):
    """concentration (ppm) giving fatality probability P after t_min minutes"""
    a, b, n = PROBIT_H2S[model or PROBIT_MODEL]
    lnC = ((norm.ppf(P) + 5.0 - a) / b - math.log(t_min)) / n
    return math.exp(lnC) / MG_PER_PPM_H2S

# --------------------------------------------------------------------------- scenarios x weather
def main():
    WEATHER = {'D5': ('D', 5.0), 'D3': ('D', 3.0), 'F2': ('F', 2.0), 'calm': (None, 0.5)}
    RATES = {  # release type: (Q m3/h instantaneous, Q m3/h time-mean, height m, note)
        'R1 chatter (valve open)': (141.0, 141.0, 18.25, 'valve at minimum stable opening, 8-s pulses'),
        'R1 chatter (episode mean)': (44.0, 44.0, 18.25, 'time-mean over the 26-s cycle'),
        'R2 downstream stop': (141.0, 84.5, 18.25, 'production through the valve, chattering'),
        'R3 isolated digester': (141.0, 41.2, 18.25, 'one digester, q/2'),
        'R5 blocked PVRV, continuous': (41.2, 41.2, 16.5, 'roof seal, no stack'),
    }
    rows = []
    for name, (Qi, Qm, h, note) in RATES.items():
        for wname, (cls, u) in WEATHER.items():
            for label, Qh in (('instantaneous', Qi), ('time-mean', Qm)):
                Q = Qh / 3600
                if cls is None:
                    z = np.linspace(0.3, 30, 300); c, b = mtt_plume(Q, z)
                    xl = float(z[np.argmax(c < LFL)]) if (c < LFL).any() else 30.0
                    xi = float(z[np.argmax(c * yH2S * 1e6 < IDLH_H2S)]) if (c * yH2S * 1e6 < IDLH_H2S).any() else 30.0
                    c1 = float(mtt_plume(Q, 1.0)[0]); rad = float(mtt_plume(Q, xl)[1])
                    rows.append(dict(release=name, Q_m3h=Qh, basis=label, weather=wname, stability='calm', u_ms=u, LFL_dist_m=xl, LFL_radius_m=rad, IDLH_dist_m=xi,
                                     C_1m_volfrac=c1, H2S_1m_ppm=c1 * yH2S * 1e6, H2S_3m_ppm=float(mtt_plume(Q, 3.0)[0]) * yH2S * 1e6, direction='vertical (buoyant plume)'))
                else:
                    xl = dist_to(LFL, Q, u, cls); xi = dist_to(IDLH_H2S / (yH2S * 1e6), Q, u, cls); xa = dist_to(AEGL3_10MIN / (yH2S * 1e6), Q, u, cls)
                    c1 = float(plume_centreline(Q, u, cls, 1.0)); c3 = float(plume_centreline(Q, u, cls, 3.0))
                    sy1, sz1, _ = plume_sigma(Q, u, cls, xl + 1e-3)
                    rows.append(dict(release=name, Q_m3h=Qh, basis=label, weather=wname, stability=cls, u_ms=u, LFL_dist_m=xl, LFL_radius_m=float(0.5 * (sy1 + sz1)) if xl > 0 else 0.0,
                                     IDLH_dist_m=xi, AEGL3_dist_m=xa, C_1m_volfrac=c1, H2S_1m_ppm=c1 * yH2S * 1e6, H2S_3m_ppm=c3 * yH2S * 1e6, direction='downwind (bent-over plume)'))
    ed = pd.DataFrame(rows)
    # puff of the failed roof seal (R5): 39 m3 at roof level, hemisphere r0 = 2.65 m
    V0 = _props_all['r5']['puff_m3']; r0 = (3 * V0 / (2 * math.pi)) ** (1 / 3)
    TAU_SEAL = 30.0          # s: assumed duration over which the compressed headspace escapes through the failed seal
    puff_rows = []
    for wname, (cls, u) in WEATHER.items():
        Qt = V0 / TAU_SEAL                                      # transient release rate, m3/s
        if cls is None:
            z = np.linspace(0.3, 60, 600); c, b = mtt_plume(Qt, z); xl_t = float(z[np.argmax(c < LFL)]) if (c < LFL).any() else 60.0
        else:
            xl_t = dist_to(LFL, Qt, u, cls, xmax=500.0)
        cls_ = cls or 'F'; t = np.linspace(0.1, 300, 3000); c, x = puff_peak(V0, u, cls_, t, r0)
        i = np.argmax(c < LFL); x_lfl = float(x[i]) if i > 0 else 0.0
        puff_rows.append(dict(weather=wname, u_ms=u, V0_m3=V0, r0_m=r0, Q_transient_m3h=Qt * 3600, LFL_dist_transient_m=xl_t,
                              LFL_reach_instant_puff_m=x_lfl + r0, time_above_LFL_s=float(t[i]) if i > 0 else 0.0))
    pf = pd.DataFrame(puff_rows)
    ed.to_csv(os.path.join(OUT, 'effect_distances.csv'), index=False, float_format='%.3f'); pf.to_csv(os.path.join(OUT, 'puff_R5.csv'), index=False, float_format='%.3f')
    pd.set_option('display.width', 250)
    print(ed[ed.basis == 'instantaneous'][['release', 'Q_m3h', 'weather', 'LFL_dist_m', 'LFL_radius_m', 'IDLH_dist_m', 'H2S_1m_ppm', 'H2S_3m_ppm']].round(2).to_string(index=False))
    print(pf.round(2).to_string(index=False))

    # H2S fatality probability at 1 m and 3 m for typical exposure times
    tab = []
    for model in ('rivm2018', 'bevi2009'):
        for t in (1, 2, 5, 10, 30):
            for C in (100, 200, 300, 500, 1000, 2500):
                tab.append(dict(model=model, t_min=t, C_ppm=C, P_fatal=float(h2s_probit_fatality(C, t, model))))
    ht = pd.DataFrame(tab).pivot(index=['model', 'C_ppm'], columns='t_min', values='P_fatal'); ht.to_csv(os.path.join(OUT, 'h2s_probit_table.csv'), float_format='%.4g'); print(ht.round(4))
    lc = pd.DataFrame([dict(model=m, t_min=t, LC01_ppm=h2s_lc(0.01, t, m), LC50_ppm=h2s_lc(0.5, t, m)) for m in ('rivm2018', 'bevi2009') for t in (1, 5, 10, 30, 60)])
    lc.to_csv(os.path.join(OUT, 'h2s_lc_table.csv'), index=False, float_format='%.4g'); print(lc.round(1).to_string(index=False))

    # --------------------------------------------------------------------------- figure
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    C = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw=dict(wspace=0.3))
    x = np.logspace(-1, 1.7, 300)
    for i, (wname, (cls, u)) in enumerate(WEATHER.items()):
        if cls is None: continue
        for Qh, ls in ((141.0, '-'), (44.0, '--')):
            ax[0].plot(x, plume_centreline(Qh / 3600, u, cls, x) * 100, color=C[i], ls=ls, label=f'{wname}, {Qh:.0f} m³/h')
    ax[0].axhline(LFL * 100, color='k', lw=0.8, ls=':'); ax[0].text(0.12, LFL * 100 * 1.1, f'LFL {LFL*100:.1f} vol %', fontsize=8)
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_ylim(0.01, 100); ax[0].set_xlabel('downwind distance from the vent (m)'); ax[0].set_ylabel('biogas on the plume axis (vol %)')
    ax[0].set_title('(a) continuous release at the PVRV (bent-over plume)', loc='left', fontweight='bold'); ax[0].legend(fontsize=7, ncol=2)
    for i, (wname, (cls, u)) in enumerate(WEATHER.items()):
        if cls is None: continue
        for Qh, ls in ((141.0, '-'), (41.2, '--')):
            ax[1].plot(x, plume_centreline(Qh / 3600, u, cls, x) * yH2S * 1e6, color=C[i], ls=ls, label=f'{wname}, {Qh:.0f} m³/h')
    for v, nm in ((IDLH_H2S, 'IDLH 100'), (AEGL3_10MIN, 'AEGL-3 (10 min) 76'), (ERPG2_H2S, 'ERPG-2 30')):
        ax[1].axhline(v, color='k', lw=0.6, ls=':'); ax[1].text(20, v * 1.1, nm, fontsize=7)
    ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_ylim(1, 5000); ax[1].set_xlabel('downwind distance (m)'); ax[1].set_ylabel('H₂S on the plume axis (ppm)')
    ax[1].set_title('(b) H₂S (2,527 ppm in the gas)', loc='left', fontweight='bold'); ax[1].legend(fontsize=7, ncol=2)
    z = np.linspace(0.3, 20, 200)
    for Qh, ls in ((141.0, '-'), (44.0, '--'), (41.2, ':')):
        c, b = mtt_plume(Qh / 3600, z); ax[2].plot(c * 100, z, color=C[4], ls=ls, label=f'{Qh:.0f} m³/h')
    ax[2].axvline(LFL * 100, color='k', lw=0.8, ls=':'); ax[2].set_xscale('log'); ax[2].set_xlim(0.1, 100); ax[2].set_xlabel('biogas on the plume axis (vol %)'); ax[2].set_ylabel('height above the vent (m)')
    ax[2].set_title('(c) calm air: buoyant plume rises vertically', loc='left', fontweight='bold'); ax[2].legend(fontsize=7)
    fig.savefig(os.path.join(FIG, 'consequence.png'), dpi=150, bbox_inches='tight'); plt.close(fig); print('fig ok')

if __name__ == '__main__':
    main()
