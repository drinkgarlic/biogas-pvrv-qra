# -*- coding: utf-8 -*-
"""
Far-field Gaussian plume for the H2S endpoints that lie beyond the roof (AEGL / IDLH / odour), and a model comparison.

Own implementation (MIT): Pasquill-Gifford sigmas (Briggs rural, the same set as consequence.py), Briggs buoyant plume rise with
final rise, ground reflection, receptor at any height. Far-field concentrations are 10-min averages, so the time-mean release
rate is used (the valve pulses at 141 m3/h with a duty cycle Q_mean / 141; see risk.py for the near-field dose).

Cross-checks (optional, only if the repositories were cloned into ../external by colab/get_external_models.sh):
  * koaandmud/GPM                     - EPA ISC3 Pasquill-Gifford sigma fits (Turner curves), Manchester teaching code
  * gerritjandebruin/gaussian-plume   - Dutch 'farm' / 'NOGEPA' sigma scheme (KNMI-type), EU CLINSH project, Zenodo DOI
Neither carries an explicit licence, so they are not vendored.
Outputs: out/farfield_h2s.csv, out/model_comparison.csv, fig/farfield_h2s.png, fig/model_comparison.png
"""
import os, sys, math, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..')); sys.path.insert(0, HERE)
import consequence as cq
OUT = os.path.join(HERE, 'out'); FIG = os.path.join(HERE, 'fig'); os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
EXT = os.path.join(ROOT, 'external')
G = 9.80665
H2S_LEVELS = {'AEGL-3 (10 min) 76 ppm': 76.0, 'IDLH 100 ppm': 100.0, 'AEGL-2 (10 min) 41 ppm': 41.0, 'AEGL-1 (10 min) 0.75 ppm': 0.75, 'odour threshold 0.01 ppm': 0.01}

def briggs_rise(x, Q, u, cls, T_gas=308.15, T_amb=288.15):
    """buoyant plume rise (m) after Briggs: transitional rise 1.6 F^(1/3) x^(2/3) / u, capped at the final rise"""
    F = G * Q / math.pi * (cq.RHO_A - cq.RHO_G) / cq.RHO_A      # buoyancy flux parameter (m4/s3): g V (rho_a - rho_g)/rho_a / pi
    F = max(F, 1e-6)
    if cls in ('A', 'B', 'C', 'D'):
        xf = 49 * F ** (5 / 8) if F < 55 else 119 * F ** 0.4
        dh_final = 1.6 * F ** (1 / 3) * xf ** (2 / 3) / u
    else:
        s = 0.020 * G / T_amb if cls == 'E' else 0.035 * G / T_amb
        dh_final = 2.6 * (F / (u * s)) ** (1 / 3)
    dh = 1.6 * F ** (1 / 3) * np.asarray(x, float) ** (2 / 3) / u
    return np.minimum(dh, dh_final)

def gaussian(Q, u, cls, x, z_rec, h_s, y=0.0, rise=True, r0=0.1):
    """volume fraction at (x, y, z_rec) for a continuous release Q (m3/s) from height h_s with plume rise and ground reflection"""
    x = np.asarray(x, float)
    sy, sz = cq.sig(np.maximum(x, 1e-3), cls)
    sy = np.sqrt(sy ** 2 + (r0 / 2) ** 2); sz = np.sqrt(sz ** 2 + (r0 / 2) ** 2)
    H = h_s + (briggs_rise(x, Q, u, cls) if rise else 0.0)
    c = Q / (2 * math.pi * u * sy * sz) * np.exp(-y ** 2 / (2 * sy ** 2)) * (np.exp(-(z_rec - H) ** 2 / (2 * sz ** 2)) + np.exp(-(z_rec + H) ** 2 / (2 * sz ** 2)))
    return np.minimum(1.0, c)

def dist_to_level(level_ppm, Q, u, cls, z_rec, h_s, yH2S, xmax=3000.0):
    """downwind distance (m) where the H2S concentration on the plume axis at height z_rec falls below level_ppm; 0 if never above"""
    x = np.logspace(-1, math.log10(xmax), 4000); c = gaussian(Q, u, cls, x, z_rec, h_s) * yH2S * 1e6
    above = c >= level_ppm
    if not above.any(): return 0.0
    return float(x[np.where(above)[0][-1]])

# --------------------------------------------------------------------------- external implementations (optional)
def ext_gpm_sigmas(cls, x):
    p = os.path.join(EXT, 'GPM')
    if not os.path.exists(os.path.join(p, 'calc_sigmas.py')): return None
    sys.path.insert(0, p); from calc_sigmas import calc_sigmas; sys.path.pop(0)
    cat = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}[cls]
    sy, sz = calc_sigmas(cat, np.asarray(x, float))
    return sy, sz

def ext_gplume_sigmas(cls, x, u, z0=0.2):
    """TNO 'farm' scheme, called with the package's own conventions (checked in dispersion_check.py): z0 is the ROUGHNESS LENGTH
    (its docstring says 'source height', but its examples pass z0 = 0.2 next to a separate hs), tc = calculate_tc(dx, u) = dx/(3600 u).
    sigma_y grows with travel time^0.35, i.e. it is a relative-diffusion (instantaneous plume) parameterisation, not a 10-min average"""
    p = os.path.join(EXT, 'gaussian-plume', 'src')
    if not os.path.exists(os.path.join(p, 'GaussianPlume', 'GPFunctions.py')): return None
    sys.path.insert(0, p); sys.path.insert(0, os.path.join(p, 'GaussianPlume'))      # the package uses bare intra-package imports
    import GPFunctions as GPF, GPConstants as GPC
    sys.path.pop(0); sys.path.pop(0)
    dc = GPC.dispersion_constants('farm'); idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}[cls]
    x = np.asarray(x, float); tc = GPF.calculate_tc(x, u)
    sy, sz = GPF.calculate_sigma(x, z0, tc, dc, idx)
    return sy, sz

def gaussian_from_sigmas(Q, u, sy, sz, z_rec, h_s):
    """the same elevated-source formula as gaussian(), for sigmas supplied by an external scheme"""
    return np.minimum(1.0, Q / (2 * math.pi * u * sy * sz) * (np.exp(-(z_rec - h_s) ** 2 / (2 * sz ** 2)) + np.exp(-(z_rec + h_s) ** 2 / (2 * sz ** 2))))

def main():
    yH2S = cq.yH2S; h_s = 18.25
    # ---- far-field H2S distances (time-mean rates, 10-min averages), receptor at roof height (18.25 m) and at ground level (1.5 m)
    rows = []
    for rel, Qh in (('R1 chatter episode (time-mean 44 m³/h)', 44.0), ('R2 downstream stop (84 m³/h)', 84.5), ('R3 isolated digester (41 m³/h)', 41.2), ('valve fully at min. opening (141 m³/h)', 141.0)):
        for wname, (cls, u) in (('D5', ('D', 5.0)), ('D3', ('D', 3.0)), ('F2', ('F', 2.0))):
            for zname, z in (('roof (18.25 m)', 18.25), ('ground (1.5 m)', 1.5)):
                r = dict(release=rel, Q_m3h=Qh, weather=wname, receptor=zname)
                for name, lv in H2S_LEVELS.items(): r[name] = dist_to_level(lv, Qh / 3600, u, cls, z, h_s, yH2S)
                rows.append(r)
    ff = pd.DataFrame(rows); ff.to_csv(os.path.join(OUT, 'farfield_h2s.csv'), index=False, float_format='%.1f')
    pd.set_option('display.width', 250); print(ff.round(1).to_string(index=False))

    # ---- model comparison at the roof height, 141 m3/h, D5 and F2
    xs = np.array([1, 2, 3, 5, 10, 30, 100, 300], float); comp = []
    for wname, (cls, u) in (('D5', ('D', 5.0)), ('F2', ('F', 2.0))):
        Q = 141.0 / 3600
        near = cq.plume_centreline(Q, u, cls, xs)
        gau = gaussian(Q, u, cls, xs, h_s, h_s, rise=False); gau_r = gaussian(Q, u, cls, xs, h_s, h_s, rise=True)
        d = {'x_m': xs, 'near-field integral plume (this work)': near * 1e2, 'Gaussian PG Briggs-rural, no rise (this work)': gau * 1e2, 'Gaussian PG + Briggs rise (this work)': gau_r * 1e2}
        s = ext_gpm_sigmas(cls, xs)                                   # same formula for every scheme (an earlier version used Q/(pi u sy sz) here: 2x)
        if s is not None:
            sy, sz = s; d['Gaussian ISC3 PG (koaandmud/GPM)'] = gaussian_from_sigmas(Q, u, np.maximum(sy, 1e-3), np.maximum(sz, 1e-3), h_s, h_s) * 1e2
        s = ext_gplume_sigmas(cls, xs, u)
        if s is not None:
            sy, sz = s; d["Gaussian TNO 'farm' scheme, z0 0.2 m (gerritjandebruin/gaussian-plume)"] = gaussian_from_sigmas(Q, u, np.maximum(sy, 1e-3), np.maximum(sz, 1e-3), h_s, h_s) * 1e2
        cdf = pd.DataFrame(d); cdf.insert(0, 'weather', wname); comp.append(cdf)
    comp = pd.concat(comp, ignore_index=True); comp.to_csv(os.path.join(OUT, 'model_comparison.csv'), index=False, float_format='%.4g')
    print('\nbiogas on the plume axis (vol %), 141 m3/h, receptor at the vent height:'); print(comp.round(4).to_string(index=False))

    # ---- figures
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'axes.titleweight': 'bold'})
    C = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    x = np.logspace(-0.3, 3.3, 400)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(wspace=0.28))
    for i, (wname, (cls, u)) in enumerate((('D5', ('D', 5.0)), ('D3', ('D', 3.0)), ('F2', ('F', 2.0)))):
        for Qh, ls in ((141.0, '-'), (44.0, '--')):
            ax[0].plot(x, gaussian(Qh / 3600, u, cls, x, 18.25, h_s) * yH2S * 1e6, color=C[i], ls=ls, label=f'{wname}, {Qh:.0f} m³/h, roof height')
        ax[1].plot(x, gaussian(84.5 / 3600, u, cls, x, 1.5, h_s) * yH2S * 1e6, color=C[i], label=f'{wname}, 84 m³/h, ground level')
    for a in ax:
        for name, lv in H2S_LEVELS.items():
            a.axhline(lv, color='k', lw=0.5, ls=':'); a.text(1.2, lv * 1.15, name, fontsize=6.5, color='#333333')
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('downwind distance (m)'); a.set_ylabel('H₂S, 10-min mean on the plume axis (ppm)'); a.legend(fontsize=7)
    ax[0].set_ylim(1e-3, 3e3); ax[1].set_ylim(1e-4, 1e2)
    ax[0].set_title('(a) receptor at the vent height: the valve is the hazard', loc='left'); ax[1].set_title('(b) ground level: odour reaches hundreds of metres, AEGL-1 tens of metres', loc='left')
    fig.savefig(os.path.join(FIG, 'farfield_h2s.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(wspace=0.28))
    for a, wname in zip(ax, ('D5', 'F2')):
        s = comp[comp.weather == wname]
        for j, col in enumerate([c for c in s.columns if c not in ('weather', 'x_m')]):
            a.plot(s['x_m'], s[col], color=C[j], marker='o', ms=3, label=col)
        a.axhline(cq.LFL * 100, color='k', lw=0.8, ls=':'); a.text(1.1, cq.LFL * 100 * 1.15, 'LFL', fontsize=7)
        a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(1e-4, 100); a.set_xlabel('downwind distance (m)'); a.set_ylabel('biogas on the axis (vol %)'); a.legend(fontsize=6.5); a.set_title(f'{wname}: 141 m³/h, receptor at the vent height', loc='left')
    fig.savefig(os.path.join(FIG, 'model_comparison.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    print('figs ok')

if __name__ == '__main__':
    main()
