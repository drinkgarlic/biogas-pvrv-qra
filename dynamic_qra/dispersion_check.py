# -*- coding: utf-8 -*-
"""
Consistency check of the dispersion models used or pulled from GitHub, before any of them is trusted in the QRA.

What is checked
  1. The sigma schemes themselves: Briggs (1973) open-country (this work), the EPA ISC3 fits of the Pasquill-Gifford-Turner
     curves (koaandmud/GPM) and the TNO 'farm' scheme (gerritjandebruin/gaussian-plume), at 10 m - 1 km for classes D and F,
     against the tabulated PG-Turner values that every textbook reproduces.
  2. The concentration formula: all three sigma schemes run through ONE elevated-source Gaussian with ground reflection
     (the earlier model_comparison.csv evaluated the external schemes with the ground-level form Q/(pi u sy sz), i.e. 2x).
  3. Mass balance: the integral of u C over the half-space must return Q.
  4. The near-field integral plume: the 1/pi (fully reflected) form used so far against the reflected form with the vent
     outlet h_v above the roof, which tends to the free plume (1/2pi) when sigma_z << h_v.
  5. Unit check of the external package's own concentration routine against the same formula.
  6. Density: when would a dense-gas model (HEGADAS / HeavyGas.jl) apply at all.
Outputs: out/dispersion_check_*.csv, fig/dispersion_check.png
"""
import os, sys, math
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..')); sys.path.insert(0, HERE)
import consequence as cq, gaussian_farfield as gf
OUT = os.path.join(HERE, 'out'); FIG = os.path.join(HERE, 'fig'); os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
EXT = os.path.join(ROOT, 'external')
Z0 = 0.2            # roughness length (m) for the TNO scheme: open site with low buildings; the package's own examples use 0.2
H_S = 18.25         # vent height above grade (m)
H_V = 1.0           # vent outlet above the digester roof (m)
Q = 141.0 / 3600    # m3/s at the minimum stable opening

# PG-Turner reference values (Turner 1970 workbook curves, as tabulated in Seinfeld & Pandis / de Nevers), metres
PG_REF = {('D', 100): (8.1, 4.6), ('D', 1000): (68.0, 32.0), ('F', 100): (4.0, 2.2), ('F', 1000): (34.0, 14.0)}

def ext_gpm(cls, x):
    p = os.path.join(EXT, 'GPM')
    if not os.path.exists(os.path.join(p, 'calc_sigmas.py')): return None
    sys.path.insert(0, p); from calc_sigmas import calc_sigmas; sys.path.pop(0)
    return calc_sigmas({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}[cls], np.asarray(x, float))

def ext_farm(cls, x, u, z0=Z0, mode='farm'):
    """the package's own convention: z0 = roughness length (its docstring says source height, its examples pass z0 = 0.2 next to hs),
    tc = calculate_tc(dx, u) = dx / (3600 u), i.e. travel time in hours (the author left a 'TODO: is this correct?')"""
    p = os.path.join(EXT, 'gaussian-plume', 'src')
    if not os.path.exists(os.path.join(p, 'GaussianPlume', 'GPFunctions.py')): return None
    sys.path.insert(0, p); sys.path.insert(0, os.path.join(p, 'GaussianPlume'))
    import GPFunctions as GPF, GPConstants as GPC
    sys.path.pop(0); sys.path.pop(0)
    dc = GPC.dispersion_constants(mode); idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}[cls]
    x = np.asarray(x, float); tc = GPF.calculate_tc(x, u)
    return GPF.calculate_sigma(x, z0, tc, dc, idx)

def ext_ch4disp(cls, x):
    """yunhal/CH4_dispersion (MIT): its own transcription of the ISC3 PG fits; distance in km"""
    p = os.path.join(EXT, 'CH4_dispersion', 'code')
    if not os.path.exists(os.path.join(p, 'gaussian_plume_functions.py')): return None
    import ast                                   # the module imports pyproj / xarray at the top; take only this function
    tree = ast.parse(open(os.path.join(p, 'gaussian_plume_functions.py'), encoding='utf-8').read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_sigma_vals')
    ns = {'np': np}; exec(compile(ast.Module(body=[fn], type_ignores=[]), 'compute_sigma_vals', 'exec'), ns)
    xk = np.asarray(x, float) / 1000.0
    return ns['compute_sigma_vals'](np.array([cls]), xk)

def fgp_coefficients():
    """Hammerling-Research-Group/FastGaussianPuff (MIT, C++): read the ISC3 coefficients from getSigmaCoefficients() and compare
    them number by number with koaandmud/GPM; the C++ needs Eigen + pybind11 to build, so it is not run"""
    f = os.path.join(EXT, 'FastGaussianPuff', 'src', 'CGaussianPuff.cpp')
    g = os.path.join(EXT, 'GPM', 'calc_sigmas.py')
    if not (os.path.exists(f) and os.path.exists(g)): return None
    import re
    src = open(f, encoding='utf-8').read(); src = src[src.index('void getSigmaCoefficients'):]
    num = lambda t: sorted(set(round(float(v), 4) for v in re.findall(r'(?<![\w.])\d+\.\d+', t)))
    a, b = num(src), num(open(g, encoding='utf-8').read())
    return sorted(set(b) - set(a)), sorted(set(a) - set(b))

def gauss_axis(sy, sz, u, z_rec=H_S, h=H_S):
    """elevated source, receptor on the axis at z_rec, ground reflection: the one formula used for every scheme here"""
    return np.minimum(1.0, Q / (2 * math.pi * u * sy * sz) * (np.exp(-(z_rec - h) ** 2 / (2 * sz ** 2)) + np.exp(-(z_rec + h) ** 2 / (2 * sz ** 2))))

def mass_balance(sy, sz, u, h=H_S, images=2):
    """integral of u C dy dz over y in R, z >= 0, divided by Q"""
    y = np.linspace(-8 * sy, 8 * sy, 801); z = np.linspace(0, h + 8 * sz, 1601); Y, Z = np.meshgrid(y, z)
    c = Q / (2 * math.pi * u * sy * sz) * np.exp(-Y ** 2 / (2 * sy ** 2)) * (np.exp(-(Z - h) ** 2 / (2 * sz ** 2)) + (np.exp(-(Z + h) ** 2 / (2 * sz ** 2)) if images == 2 else 0))
    return float(np.trapezoid(np.trapezoid(c, y, axis=1), z) * u / Q)

def main():
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    xs = np.array([10, 30, 100, 300, 1000], float)
    # ---- 1. sigma schemes
    rows = []
    for cls, u in (('D', 5.0), ('F', 2.0)):
        b = cq.sig(xs, cls); g = ext_gpm(cls, xs); f = ext_farm(cls, xs, u); f_wrong = ext_farm(cls, xs, u, z0=H_S)
        for i, x in enumerate(xs):
            r = dict(cls=cls, x_m=x, sy_Briggs=b[0][i], sz_Briggs=b[1][i])
            if g is not None: r.update(sy_ISC3=g[0][i], sz_ISC3=g[1][i])
            if f is not None: r.update(sy_farm_z0_02=f[0][i], sz_farm_z0_02=f[1][i], sy_farm_z0_18=f_wrong[0][i], sz_farm_z0_18=f_wrong[1][i])
            ref = PG_REF.get((cls, int(x)));  r.update(sy_PG_table=ref[0] if ref else np.nan, sz_PG_table=ref[1] if ref else np.nan)
            rows.append(r)
    for r in rows:
        c = ext_ch4disp(r['cls'], [r['x_m']])
        if c is not None: r['sy_CH4disp'], r['sz_CH4disp'] = float(c[0][0]), float(c[1][0])
    sg = pd.DataFrame(rows); sg.to_csv(os.path.join(OUT, 'dispersion_check_sigmas.csv'), index=False, float_format='%.3g')
    print('1. sigma schemes (m)'); print(sg.round(3).to_string(index=False))

    fg = fgp_coefficients()
    if fg is not None:
        print(f'   FastGaussianPuff vs GPM coefficient sets: only in GPM {fg[0]}, only in FastGaussianPuff {fg[1]}')

    # ---- 2. one concentration formula, three sigma schemes, plus the near-field plume
    xs2 = np.array([1, 2, 3, 5, 10, 30, 100, 300, 1000], float); rows = []
    for cls, u in (('D', 5.0), ('F', 2.0)):
        b = cq.sig(np.maximum(xs2, 1e-3), cls); g = ext_gpm(cls, xs2); f = ext_farm(cls, xs2, u)
        near = cq.plume_centreline(Q, u, cls, xs2)
        for i, x in enumerate(xs2):
            r = dict(cls=cls, x_m=x, near_field_plume=near[i] * 100, gauss_Briggs=gauss_axis(b[0][i], b[1][i], u) * 100)
            if g is not None: r['gauss_ISC3'] = gauss_axis(g[0][i], g[1][i], u) * 100
            if f is not None: r['gauss_farm'] = gauss_axis(f[0][i], f[1][i], u) * 100
            rows.append(r)
    cc = pd.DataFrame(rows); cc.to_csv(os.path.join(OUT, 'dispersion_check_conc.csv'), index=False, float_format='%.4g')
    print('\n2. biogas on the axis at the vent height (vol %), 141 m3/h, same Gaussian formula, different sigmas'); print(cc.round(4).to_string(index=False))

    # ---- 3. mass balance
    rows = []
    for cls, u in (('D', 5.0), ('F', 2.0)):
        for x in (10.0, 100.0, 1000.0):
            sy, sz = cq.sig(x, cls); rows.append(dict(cls=cls, x_m=x, scheme='Briggs', sy=sy, sz=sz, int_uC_dydz_over_Q=mass_balance(sy, sz, u)))
            g = ext_gpm(cls, [x])
            if g is not None: rows.append(dict(cls=cls, x_m=x, scheme='ISC3', sy=g[0][0], sz=g[1][0], int_uC_dydz_over_Q=mass_balance(g[0][0], g[1][0], u)))
            f = ext_farm(cls, [x], u)
            if f is not None: rows.append(dict(cls=cls, x_m=x, scheme='farm', sy=f[0][0], sz=f[1][0], int_uC_dydz_over_Q=mass_balance(f[0][0], f[1][0], u)))
    mb = pd.DataFrame(rows); mb.to_csv(os.path.join(OUT, 'dispersion_check_mass.csv'), index=False, float_format='%.4g')
    print('\n3. mass balance (should be 1.000)'); print(mb.round(4).to_string(index=False))

    # ---- 4. near-field form: 1/pi (as used) vs reflected with h_v vs free plume
    rows = []
    for wname, (cls, u) in (('D5', ('D', 5.0)), ('F2', ('F', 2.0))):
        for x in (1.0, 2.0, 3.0, 5.0, 10.0):
            sy, sz, zr = cq.plume_sigma(Q, u, cls, x)
            c_pi = min(1.0, Q / (math.pi * u * sy * sz)); c_free = min(1.0, Q / (2 * math.pi * u * sy * sz))
            c_ref = min(1.0, Q / (2 * math.pi * u * sy * sz) * (1 + math.exp(-2 * (H_V + zr) ** 2 / sz ** 2)))
            rows.append(dict(weather=wname, x_m=x, sigma_y=sy, sigma_z=sz, rise_m=zr, C_1_over_pi=c_pi * 100, C_reflected_hv1m=c_ref * 100, C_free_plume=c_free * 100,
                             H2S_ppm_1_over_pi=c_pi * cq.yH2S * 1e6, H2S_ppm_reflected=c_ref * cq.yH2S * 1e6))
    nf = pd.DataFrame(rows); nf.to_csv(os.path.join(OUT, 'dispersion_check_nearfield.csv'), index=False, float_format='%.4g')
    print('\n4. near-field plume on the axis (vol %, H2S ppm): the 1/pi form used so far vs reflection with the outlet 1 m above the roof vs a free plume'); print(nf.round(3).to_string(index=False))
    # LFL distances under the two forms
    for wname, (cls, u) in (('D5', ('D', 5.0)), ('F2', ('F', 2.0))):
        x = np.logspace(-2, 2, 2000); sy = np.zeros_like(x); sz = np.zeros_like(x); zr = np.zeros_like(x)
        for i, xi in enumerate(x): sy[i], sz[i], zr[i] = cq.plume_sigma(Q, u, cls, xi)
        c_pi = Q / (math.pi * u * sy * sz); c_ref = Q / (2 * math.pi * u * sy * sz) * (1 + np.exp(-2 * (H_V + zr) ** 2 / sz ** 2))
        d_pi = x[np.argmax(c_pi < cq.LFL)]; d_ref = x[np.argmax(c_ref < cq.LFL)]
        print(f'   LFL distance {wname}: 1/pi form {d_pi:.2f} m, reflected form {d_ref:.2f} m')

    # ---- 5. unit check of the package's own concentration routine (g/s -> ppm) against the same formula
    p = os.path.join(EXT, 'gaussian-plume', 'src')
    if os.path.exists(os.path.join(p, 'GaussianPlume', 'GPFunctions.py')):
        sys.path.insert(0, p); sys.path.insert(0, os.path.join(p, 'GaussianPlume')); import GPFunctions as GPF, GPConstants as GPC; sys.path.pop(0); sys.path.pop(0)
        sy, sz = 10.0, 5.0; u = 5.0; M = 16.04; qs_gs = 1.0                       # 1 g/s CH4
        own = qs_gs / M * 0.0224 / (2 * math.pi * u * sy * sz) * (1 + math.exp(-(2 * H_S) ** 2 / (2 * sz ** 2))) * 1e6     # mol/s -> m3/s at 0 C
        pkg = float(GPF.calculate_concentration(qs_gs, u, sy, sz, 0.0, H_S, H_S, 1000.0, M))
        print(f'\n5. package calculate_concentration: {pkg:.4g} ppm vs the same formula with 22.4 L/mol: {own:.4g} ppm  (ratio {pkg / own:.3f}; the package uses {GPC.liter_per_mole_air} L/mol)')

    # ---- 6. when would a dense-gas model apply
    y = np.linspace(0.40, 0.75, 8); rho_rel = (y * 16.04 + (1 - y) * 44.01) / 28.96 * (288.15 / 308.15)
    print('\n6. relative density of the vented gas (35 C) vs CH4 fraction: ' + ', '.join(f'{yy:.2f}->{r:.2f}' for yy, r in zip(y, rho_rel)) + '  -> dense (> 1) only below ~52 % CH4 at 35 C; HEGADAS-type models need rho_g > rho_a')

    # ---- figure
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'axes.titleweight': 'bold'})
    C = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    x = np.logspace(0, 3, 300)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(wspace=0.3))
    for j, (cls, u, ls) in enumerate((('D', 5.0, '-'), ('F', 2.0, '--'))):
        b = cq.sig(x, cls); g = ext_gpm(cls, x); f = ext_farm(cls, x, u)
        ax[0].plot(x, b[0], color=C[0], ls=ls, label=f'Briggs rural {cls}'); ax[1].plot(x, b[1], color=C[0], ls=ls, label=f'Briggs rural {cls}')
        if g is not None: ax[0].plot(x, g[0], color=C[1], ls=ls, label=f'ISC3 PG fit {cls} (GPM)'); ax[1].plot(x, g[1], color=C[1], ls=ls, label=f'ISC3 PG fit {cls} (GPM)')
        if f is not None: ax[0].plot(x, f[0], color=C[2], ls=ls, label=f"TNO 'farm' {cls}, z0 0.2 m"); ax[1].plot(x, f[1], color=C[2], ls=ls, label=f"TNO 'farm' {cls}, z0 0.2 m")
        for (c2, xr), (syr, szr) in PG_REF.items():
            if c2 == cls: ax[0].plot(xr, syr, 'k*', ms=9); ax[1].plot(xr, szr, 'k*', ms=9)
        bb = cq.sig(np.maximum(x, 1e-3), cls); near = cq.plume_centreline(Q, u, cls, x)
        ax[2].plot(x, gauss_axis(bb[0], bb[1], u) * 100, color=C[0], ls=ls, label=f'Gaussian, Briggs {cls}')
        if g is not None: ax[2].plot(x, gauss_axis(g[0], g[1], u) * 100, color=C[1], ls=ls, label=f'Gaussian, ISC3 {cls}')
        if f is not None: ax[2].plot(x, gauss_axis(f[0], f[1], u) * 100, color=C[2], ls=ls, label=f"Gaussian, 'farm' {cls}")
        ax[2].plot(x, near * 100, color=C[3], ls=ls, label=f'near-field plume {cls}')
    ax[0].plot([], [], 'k*', label='PG-Turner table (100 m, 1 km)')
    for a, t, yl in zip(ax, ('(a) sigma_y', '(b) sigma_z', '(c) axis concentration at the vent height, 141 m3/h')[:3], ('σy (m)', 'σz (m)', 'biogas (vol %)')):
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('downwind distance (m)'); a.set_ylabel(yl); a.set_title(t, loc='left'); a.legend(fontsize=6.5)
    ax[2].axhline(cq.LFL * 100, color='k', lw=0.6, ls=':'); ax[2].text(1.1, cq.LFL * 100 * 1.15, 'LFL', fontsize=7); ax[2].set_ylim(1e-4, 100)
    ax[0].axvspan(1, 100, color='#eeeeee', zorder=0); ax[1].axvspan(1, 100, color='#eeeeee', zorder=0); ax[0].text(1.3, 20, 'PG curves not defined\nbelow 100 m', fontsize=7, color='#555555')
    fig.savefig(os.path.join(FIG, 'dispersion_check.png'), dpi=150, bbox_inches='tight'); plt.close(fig); print('fig ok')

if __name__ == '__main__':
    main()
