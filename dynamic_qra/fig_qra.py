# -*- coding: utf-8 -*-
"""Dynamic QRA figures: event tree, IR by case and scenario, monthly / hour-of-day profiles, generic vs dynamic, tornado, Bayesian updating."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'legend.fontsize': 8, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold'})
C = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#17becf', '#7f7f7f']; INK = '#111111'
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'out'); FIG = os.path.join(HERE, 'fig'); os.makedirs(FIG, exist_ok=True)
r = pd.read_csv(os.path.join(OUT, 'risk_summary.csv')); prof = pd.read_csv(os.path.join(OUT, 'risk_profiles.csv'))
sens = pd.read_csv(os.path.join(OUT, 'risk_sensitivity.csv'))
labels = {(1, 1.0): 'Case 1 (4.0 kPag)\nflare start 100 %', (1, 0.8): 'Case 1 (4.0 kPag)\nflare start 80 %', (3, 1.0): 'Case 3 (2.0 kPag)\nflare start 100 %', (3, 0.8): 'Case 3 (2.0 kPag)\nflare start 80 %'}
base = r['case'].astype(str).str.fullmatch(r'\d+'); rr = r[base].copy(); rr['case'] = rr['case'].astype(int)
opts = r[~base & (r.case != 'generic')].copy()      # design options: '<case> + <option>'
order = [(1, 1.0), (1, 0.8), (3, 1.0), (3, 0.8)]
BR = [('IR_resp_demand', 'ignition failure (chatter)'), ('IR_resp_stop', 'downstream stop'), ('IR_resp_foam', 'foam blockage'), ('IR_resp_block', 'condensate / freezing / XV'), ('IR_resp_wv', 'vent valve stuck'), ('IR_resp_R5', 'blocked PVRV -> seal failure'), ('IR_rout', 'routine presence')]
BC = [C[1], C[0], C[2], C[4], C[3], C[5], C[7]]

# ---------------------------------------------------------------- 1. event tree
fig, ax = plt.subplots(figsize=(13, 5.6)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 60)
def box(x, y, w, h, t, fc='#f4f6fa', fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4', fc=fc, ec='#555555', lw=0.8)); ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs)
def arr(x0, y0, x1, y1, t='', dy=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, color=INK, lw=0.8))
    if t: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, t, ha='center', fontsize=7.2, color='#333333')
box(1, 44, 17, 12, 'BN (Appendix V)\nP(open | hour, initiating event)\n+ P(operator arrives\nwhile venting)', fc='#e3f2fd')
box(22, 44, 16, 12, 'release type\nR1 chatter 141 m³/h pulses\nR2/R3 continuous 41-84 m³/h\nR5 seal failure (39 m³ + q/2)', fc='#fff3e0')
box(42, 44, 14, 12, 'weather (day/night)\nD5 / D3 / F2 / calm\nplume: LFL 1-3 m,\nH₂S 200-1,400 ppm at 1 m', fc='#fff3e0')
box(59, 50, 16, 8, 'detection -> action -> roof?\nflare alarm 0.1 / 0.05\nPAH (PT-101) 0.7 / 0.5\nno signal (WV) 0.3 / 0.2', fc='#ffebee', fs=7.2); box(59, 40, 16, 7, 'routine roof work\n2 h/week (day), 0.3 near\nthe valve', fc='#ffebee', fs=7.2)
box(78, 52, 20, 6, 'inside flammable plume (0.1)\n× ignition (0.02 · overlap + 0.002)\n× fatality in small plume (0.1)', fc='#e8f5e9', fs=7.2)
box(78, 44, 20, 6, 'prolonged stay in H₂S plume (0.02)\n× probit(C at 1 m, 5 min)', fc='#e8f5e9', fs=7.2)
box(78, 33, 20, 8, 'R5 blocked PVRV: 28.5 min PAH → failure\nR(≤0.5 h) arrives; manual vent on the roof,\nfouled 0.3, 3 min to open → prevent /\nbeside the vent 15 min / on roof at failure', fc='#e8f5e9', fs=6.8)
arr(18, 50, 22, 50); arr(38, 50, 42, 50); arr(56, 50, 59, 54); arr(56, 50, 59, 43.5); arr(75, 54, 78, 55); arr(75, 54, 78, 47); arr(75, 43.5, 78, 47); arr(75, 43.5, 78, 37)
box(1, 10, 97, 20, '', fc='#fafafa')
ax.text(49.5, 26, 'risk integration, hour by hour', ha='center', fontsize=9, fontweight='bold')
ax.text(49.5, 20.5, r'$PLL(h)=\sum_k o_k(h)\,P_{roof,k}(h)\,P_{fat}(k,w(h)) + \sum_k s_k(h)\,f_{rout}(h)\,p_{near}\,P_{fat,hour}(k,w(h)) + P_{init,R5}(h)\,[P_{prev}P_{fat,vent} + P_{on\,roof}(P_{seal}+P_{fat,cloud})], \quad IR_{worker}=PLL_{resp}/3+PLL_{rout}$', ha='center', fontsize=8.6)
ax.text(49.5, 14.5, 'o_k: P(operator finds the valve venting);  s_k: state probability (valve open);  f_rout: 2 h/week of daytime roof work\n-> PLL and IR, monthly and hour-of-day profiles, LSIR at 3 m, CH₄ emission, design options A / B', ha='center', fontsize=8, color='#333333')
ax.text(1, 4, 'Blue: frequency side from the BN.  Orange: consequence side (source term, weather, plume).  Red: exposure.  Green: fatality probability per exposure.  Values in brackets are the base-case assumptions (tornado in fig. 5).', fontsize=7.5, color='#444444')
fig.savefig(os.path.join(FIG, 'qra_event_tree.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 2. IR by case, stacked by scenario + generic
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(wspace=0.35, width_ratios=[1.5, 1]))
x = np.arange(4); bottom = np.zeros(4)
for (col, name), colr in zip(BR, BC):
    v = np.array([float(rr[(rr.case == c) & (np.isclose(rr.f_on, f))][col].iloc[0]) for c, f in order]); ax[0].bar(x, v, 0.6, bottom=bottom, color=colr, label=name); bottom += v
for xi, t in zip(x, bottom): ax[0].text(xi, t * 1.08, f'{t:.1e}', ha='center', fontsize=8)
iw = np.array([float(rr[(rr.case == c) & (np.isclose(rr.f_on, f))]['IR_worker'].iloc[0]) for c, f in order]); ax[0].plot(x, iw, 'kv', ms=7, label='IR, most exposed individual (1/3 of alarms + routine)')
for xi, t in zip(x, iw): ax[0].text(xi + 0.32, t, f'{t:.1e}', ha='left', va='center', fontsize=7.5)
g = float(r[r.case == 'generic']['IR_resp'].iloc[0]); ax[0].axhline(g, color=C[7], lw=1, ls='--'); ax[0].text(3.45, g * 1.3, f'generic-frequency QRA {g:.1e}', fontsize=7.5, ha='right', color=C[7])
for v, t in ((1e-3, 'HSE: intolerable for workers above 1e-3'), (1e-6, 'broadly acceptable below 1e-6')):
    ax[0].axhline(v, color='k', lw=0.6, ls=':'); ax[0].text(-0.4, v * 0.62, t, fontsize=7, color='#333333', bbox=dict(fc='white', ec='none', alpha=0.75, pad=1))
ax[0].set_yscale('log'); ax[0].set_ylim(1e-7, 3e-2); ax[0].set_xticks(x); ax[0].set_xticklabels([labels[k] for k in order], fontsize=7.5); ax[0].set_ylabel('fatality risk per year: bars = PLL over all responders and routine work')
ax[0].set_title('(a) annual IR by case and initiating event', loc='left'); ax[0].legend(fontsize=7, ncol=2, loc='upper right')
ch4 = [float(rr[(rr.case == c) & (np.isclose(rr.f_on, f))]['ch4_kg'].iloc[0]) for c, f in order]; op = [float(rr[(rr.case == c) & (np.isclose(rr.f_on, f))]['openings'].iloc[0]) for c, f in order]
ax[1].bar(x - 0.18, op, 0.36, color=C[1], label='PVRV openings (per year)'); a2 = ax[1].twinx(); a2.bar(x + 0.18, ch4, 0.36, color=C[2], label='CH₄ vented (kg/year)'); a2.grid(False)
ax[1].set_yscale('log'); a2.set_yscale('log'); ax[1].set_xticks(x); ax[1].set_xticklabels([labels[k].replace(' kPag)', ')') for k in order], fontsize=7); ax[1].set_ylabel('openings (per year)'); a2.set_ylabel('CH₄ (kg/year)')
ax[1].set_title('(b) frequency and emission', loc='left'); h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right')
fig.savefig(os.path.join(FIG, 'qra_ir_by_case.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 3. dynamic profiles
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(wspace=0.28))
m = prof[prof.kind == 'month'].copy(); m['datetime'] = pd.to_datetime(m['datetime']); m = m[m['datetime'] < '2021-11-01']
for i, (c, f) in enumerate(order):
    s = m[(m.case == c) & (np.isclose(m.f_on, f))].sort_values('datetime'); ax[0].plot(s['datetime'], s['IR_total'], color=C[i], marker='o', ms=3, label=labels[(c, f)].replace('\n', ', '))
ax[0].set_yscale('log'); ax[0].set_ylabel('IR contribution of the month (per year)'); ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m')); ax[0].set_title('(a) month by month', loc='left'); ax[0].legend(fontsize=7)
hd = prof[prof.kind == 'hour']
for i, (c, f) in enumerate(order):
    s = hd[(hd.case == c) & (np.isclose(hd.f_on, f))].sort_values('hour'); ax[1].plot(s['hour'], s['IR_total'] * 1e3, color=C[i], marker='o', ms=3, label=labels[(c, f)].replace('\n', ', '))
ax[1].axvspan(8, 20, color='#ffe0b2', alpha=0.4, lw=0); ax[1].text(14, ax[1].get_ylim()[1] * 0.95, 'day shift 08-20', ha='center', fontsize=8, va='top')
ax[1].set_xlabel('hour of day'); ax[1].set_ylabel('IR contribution of the hour (×10⁻³ per year)'); ax[1].set_xticks(range(0, 24, 3)); ax[1].set_title('(b) hour of day', loc='left'); ax[1].legend(fontsize=7)
fig.savefig(os.path.join(FIG, 'qra_profiles.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 4. tornado (Case 1, 100 % and 80 %)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(wspace=0.6))
for a, f in zip(ax, (1.0, 0.8)):
    s = sens[(sens.case == 1) & (np.isclose(sens.f_on, f))].copy(); b = float(s['base'].iloc[0]); s['span'] = (np.log10(s['high']) - np.log10(s['low'])).abs(); s = s.sort_values('span'); y = np.arange(len(s))
    a.barh(y, np.log10(s['low']) - np.log10(b), left=np.log10(b), color=C[0], height=0.6, label='first value'); a.barh(y, np.log10(s['high']) - np.log10(b), left=np.log10(b), color=C[1], height=0.6, label='second value')
    a.axvline(np.log10(b), color=INK, lw=0.8); a.set_yticks(y); a.set_yticklabels(s['parameter'], fontsize=7.2); a.grid(axis='y', visible=False)
    ticks = np.arange(-4, -1); a.set_xticks(ticks); a.set_xlim(-4.2, -1.8); a.set_xticklabels([f'1e{t}' for t in ticks]); a.set_xlabel(f'PLL, responders + routine work (base {b:.1e})')
    a.set_title(f'Case 1, flare start {int(f*100)} %', loc='left'); a.legend(fontsize=7, loc='lower right')
fig.savefig(os.path.join(FIG, 'qra_tornado.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 5. Bayesian updating
up_path = os.path.join(OUT, 'updating.csv')
if os.path.exists(up_path):
    u = pd.read_csv(up_path)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(wspace=0.28))
    for a, f in zip(ax, (1.0, 0.8)):
        for i, tp in enumerate((0.02, 0.05, 0.10)):
            s = u[(u.case == 1) & (np.isclose(u.f_on, f)) & (np.isclose(u.true_pfd, tp))].sort_values('year')
            a.fill_between(s['year'], s['IR_lo'], s['IR_hi'], color=C[i], alpha=0.15, lw=0); a.plot(s['year'], s['IR'], color=C[i], marker='o', ms=4, label=f'records drawn with PFD {tp}, foaming {s.true_foam.iloc[0]:.0f}/yr')
        a.set_yscale('log'); a.set_xlabel('years of operating records (2,000 flare demands and the foam log per year)'); a.set_ylabel('IR of the responding operator (per year)')
        a.axhline(1e-3, color='k', lw=0.6, ls=':'); a.set_title(f'Case 1, flare start {int(f*100)} %: posterior mean and 5-95 % band (year 0 = prior)', loc='left', fontsize=9); a.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, 'qra_updating.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
print('figs ok')

# ---------------------------------------------------------------- 6. design options
fig, ax = plt.subplots(figsize=(9, 4.2))
onames = [('base', None), ('A: PVRV open indication', 'direct indication'), ('B: independent second relief', 'second relief'), ('A + B', 'both')]
w = 0.2
for j, (nm, key) in enumerate(onames):
    v = []
    for c, f in order:
        src = rr[(rr.case == c) & (np.isclose(rr.f_on, f))] if key is None else opts[(opts.case == f'{c} + {key}') & (np.isclose(opts.f_on, f))]
        v.append(float(src['IR_worker'].iloc[0]))
    ax.bar(x + (j - 1.5) * w, v, w, color=[C[7], C[0], C[2], C[1]][j], label=nm)
    for xi, t in zip(x + (j - 1.5) * w, v): ax.text(xi, t * 1.1, f'{t:.1e}', ha='center', fontsize=6.5, rotation=90)
ax.axhline(1e-3, color='k', lw=0.6, ls=':'); ax.text(3.45, 1.15e-3, 'HSE: intolerable for workers above 1e-3', fontsize=7, ha='right')
ax.set_yscale('log'); ax.set_ylim(1e-6, 1e-2); ax.set_xticks(x); ax.set_xticklabels([labels[k] for k in order], fontsize=7.5); ax.set_ylabel('IR of the most exposed individual (per year)')
ax.set_title('Design options: A) position switch / gas detector at the PVRV, B) second relief path on its own nozzle (beta = 0.1) and a remotely operated vent', loc='left', fontsize=8.5); ax.legend(fontsize=7, ncol=4, loc='upper right')
fig.savefig(os.path.join(FIG, 'qra_design_options.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
print('fig_qra ok')
