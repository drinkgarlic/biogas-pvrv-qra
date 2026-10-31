# -*- coding: utf-8 -*-
"""Figures of the research report (Korean labels): appendix VI and the matching figures of chapter IV.

    python report_figs/figs_appendix6_ko.py            # after colab/run_dynamic_qra.py (reads dynamic_qra/out)
    python report_figs/figs_appendix6_ko.py --ref      # from the reference tables in results/tables

Output: report_figs/out/*.png (6.4 in wide = 13.6 cm body width of the report). See report_figs/README.md for which
figure of the report each file is. Needs a Korean font (Noto Sans CJK KR or NanumGothic); in Colab run
`apt-get -qq install -y fonts-nanum` first."""
import os, sys, glob, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(ROOT, 'report_figs', 'out'); os.makedirs(FD, exist_ok=True)
QD = os.path.join(ROOT, 'dynamic_qra')
OUTD = os.path.join(ROOT, 'results', 'tables') if '--ref' in sys.argv or not os.path.exists(os.path.join(QD, 'out', 'risk_summary.csv')) else os.path.join(QD, 'out')
print('tables from', OUTD)
sys.path.insert(0, QD)
import consequence as cq, gaussian_farfield as gf, dispersion_check as dc
for pat in ('/usr/share/fonts/truetype/nanum/NanumGothic*.ttf', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/**/NotoSansCJK*.ttc'):
    for f in glob.glob(pat, recursive=True):
        try: fm.fontManager.addfont(f)
        except Exception: pass
import logging; logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
_have = {f.name for f in fm.fontManager.ttflist}
KFONT = next((n for n in ('Noto Sans CJK KR', 'NanumGothic', 'Malgun Gothic', 'AppleGothic') if n in _have), None)
if KFONT is None: print('WARNING: no Korean font found - install fonts-nanum (Colab: !apt-get -qq install -y fonts-nanum)')
plt.rcParams.update({'font.family': [KFONT, 'DejaVu Sans'] if KFONT else ['DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 9, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 7.5, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
                     'axes.edgecolor': '#999999', 'axes.labelcolor': '#333333', 'xtick.color': '#555555', 'ytick.color': '#555555', 'savefig.dpi': 220})
C = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#17becf', '#7f7f7f']
INK = '#111111'
def save(fig, name):
    fig.savefig(os.path.join(FD, name + '.png'), bbox_inches='tight', facecolor='white'); plt.close(fig); print('fig', name)
CASEL = {(1, 1.0): 'Case 1\n기동 100 %', (1, 0.8): 'Case 1\n기동 80 %', (3, 1.0): 'Case 3\n기동 100 %', (3, 0.8): 'Case 3\n기동 80 %'}
ORDER = [(1, 1.0), (1, 0.8), (3, 1.0), (3, 0.8)]
r = pd.read_csv(os.path.join(OUTD, 'risk_summary.csv'))
isbase = r['case'].astype(str).str.fullmatch(r'\d+'); rb = r[isbase].copy(); rb['case'] = rb['case'].astype(int); ro = r[~isbase & (r.case != 'generic')]
def pick(df, c, f, col): return float(df[(df.case == c) & np.isclose(df.f_on, f)][col].iloc[0])

# ---------------------------------------------------------------- 1. 평가 체계
def f_frame():
    """[그림 부록Ⅵ-1] 동적 정량적 위험성평가의 구성"""
    INK, SUB = '#111111', '#444444'
    BLUE, ORANGE, GREEN, RED, GRAY = '#e3f2fd', '#fff3e0', '#e8f5e9', '#ffebee', '#f5f5f5'

    fig, ax = plt.subplots(figsize=(6.4, 4.0)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 64)
    W, H = 27, 10.5
    X = {1: 3, 2: 36.5, 3: 70}                       # left edges of the three columns
    def box(col, y, title, sub, fc, w=W, h=H, x=None):
        x = X[col] if x is None else x
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec='#777777', lw=0.7))
        ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center', fontsize=8.2, fontweight='bold', color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center', fontsize=7.0, color=SUB)
    def arr(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, color=INK, lw=0.8, shrinkA=0, shrinkB=0))

    yt, ym = 50, 29                                  # top row, middle row
    box(1, yt, '방출원 (부록 Ⅳ)', '1초 압력 모델, 방출 유형 R1~R5', ORANGE)
    box(2, yt, '기상·가스 조성', '기상청 시간자료 · 시간별 조성', ORANGE)
    box(3, yt, '빈도 (부록 Ⅴ)', 'BN: 시간·초기사건별 개방확률', BLUE)
    box(1, ym, '대기확산', '근거리 플룸 + 원거리 가우시안', ORANGE)
    box(2, ym, '영향', '화재 · H₂S·CO₂ 독성 · 실 파손', GREEN)
    box(3, ym, '노출', '운전원의 감지 → 판단 → 조치', RED)

    xb, wb, yb, hb = 3, 94, 3, 15                    # bottom: integration
    ax.add_patch(FancyBboxPatch((xb, yb), wb, hb, boxstyle='round,pad=0.4,rounding_size=1.2', fc=GRAY, ec='#777777', lw=0.7))
    ax.text(50, yb + hb * 0.74, '시간별 위험도 통합 (1년 8,760 h)', ha='center', va='center', fontsize=8.6, fontweight='bold', color=INK)
    ax.text(50, yb + hb * 0.45, '위험도 = Σ (시간 h) Σ (초기사건 k)  개방확률 × 영향 × 노출', ha='center', va='center', fontsize=7.4, color=INK)
    ax.text(50, yb + hb * 0.17, '결과: PLL · 최대노출 개인위험도 (보조: CH₄ 배출량)', ha='center', va='center', fontsize=7.2, color=SUB)

    cx = {c: X[c] + W / 2 for c in X}
    arr(cx[1], yt - 0.5, cx[1], ym + H + 0.6)                         # source -> dispersion
    arr(X[2] + 3, yt - 0.5, X[1] + W - 2, ym + H + 0.6)               # weather -> dispersion
    arr(X[1] + W + 0.6, ym + H / 2, X[2] - 0.6, ym + H / 2)           # dispersion -> effect
    arr(X[2] + W + 0.6, ym + H / 2, X[3] - 0.6, ym + H / 2)           # effect -> exposure
    arr(cx[3], yt - 0.5, cx[3], ym + H + 0.6)                         # frequency -> exposure (arrival, state probability)
    arr(cx[3], ym - 0.5, cx[3], yb + hb + 0.6)                        # exposure -> integration
    arr(cx[2], ym - 0.5, cx[2], yb + hb + 0.6)                        # effect -> integration
    save(fig, 'f_frame')


# ---------------------------------------------------------------- 2. 기상
def f_met():
    m = pd.read_csv(os.path.join(OUTD, 'met_hourly.csv'), parse_dates=['datetime']).set_index('datetime')
    col = {'A': '#d62728', 'B': '#ff7f0e', 'C': '#bcbd22', 'D': '#7f7f7f', 'E': '#1f77b4', 'F': '#17becf'}
    fig = plt.figure(figsize=(6.4, 5.6)); gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, :]); hr = m.groupby(m.index.hour)['PG_class'].value_counts(normalize=True).unstack().fillna(0); bottom = np.zeros(24)
    for c in 'ABCDEF':
        if c in hr: ax.bar(hr.index, hr[c], bottom=bottom, color=col[c], label=c, width=0.85); bottom += hr[c].to_numpy()
    ax.set_xlabel('시각 (KST)'); ax.set_ylabel('시간 비율'); ax.set_title('(a) 시각별 Pasquill 안정도 등급 (Turner 방법)', loc='left'); ax.legend(ncol=6, loc='lower left', bbox_to_anchor=(0, 1.08), fontsize=7)
    ax.set_xticks(range(0, 24, 2))
    ax = fig.add_subplot(gs[1, 0]); per = m.index.to_period('M'); full = per.value_counts(); mm = m[per.isin(full[full >= 600].index)]
    mo = mm.groupby(mm.index.to_period('M'))['PG_class'].value_counts(normalize=True).unstack().fillna(0); bottom = np.zeros(len(mo))
    for c in 'ABCDEF':
        if c in mo: ax.bar(range(len(mo)), mo[c], bottom=bottom, color=col[c], width=0.85); bottom += mo[c].to_numpy()
    ax.set_xticks(range(len(mo))); ax.set_xticklabels([p.strftime('%y.%m') for p in mo.index], rotation=90, fontsize=6.5); ax.set_title('(b) 월별 안정도', loc='left'); ax.set_ylabel('시간 비율')
    ax = fig.add_subplot(gs[1, 1], projection='polar'); ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    bins = np.arange(0, 361, 22.5); calm = m['u10_ms'] < 0.5; wd = m.loc[~calm, 'wd_deg'].to_numpy() % 360; sp = m.loc[~calm, 'u10_ms'].to_numpy(); base = np.zeros(16)
    for lo, hi, cc, lab in ((0.5, 2, '#9ecae1', '0.5–2 m/s'), (2, 4, '#4292c6', '2–4 m/s'), (4, 99, '#08306b', '4 m/s 이상')):
        h, _ = np.histogram((wd[(sp >= lo) & (sp < hi)] + 11.25) % 360, bins=bins); h = h / len(m)
        ax.bar(np.radians(bins[:-1]), h, width=np.radians(22.5), bottom=base, color=cc, edgecolor='white', lw=0.3, label=lab); base += h
    ax.set_title('(c) 풍배도, 10 m', loc='left', pad=14); ax.tick_params(labelsize=6.5); ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 90, 180, 270])); ax.set_xticklabels(['N', 'E', 'S', 'W'])
    ax.legend(fontsize=6.5, loc='lower left', bbox_to_anchor=(0.82, -0.12))
    save(fig, 'f_met')

# ---------------------------------------------------------------- 3. 근거리 농도
def f_near():
    x = np.logspace(-1, math.log10(30), 300); Q = 141 / 3600
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw=dict(wspace=0.38))
    for i, (lab, cls, u) in enumerate((('D, 5 m/s', 'D', 5.0), ('D, 3 m/s', 'D', 3.0), ('F, 2 m/s', 'F', 2.0))):
        c = cq.plume_centreline(Q, u, cls, x); ax[0].plot(x, c * 100, color=C[i], label=lab); ax[1].plot(x, c * cq.yH2S * 1e6, color=C[i], label=lab)
    c, _ = cq.mtt_plume(Q, x); ax[0].plot(x, c * 100, color=C[3], ls='--', label='무풍 (연직)'); ax[1].plot(x, c * cq.yH2S * 1e6, color=C[3], ls='--', label='무풍 (연직)')
    ax[0].axhline(cq.LFL * 100, color='k', lw=0.6, ls=':'); ax[0].text(0.12, cq.LFL * 100 * 1.15, f'LFL {cq.LFL * 100:.1f} %', fontsize=7)
    for v, t in ((100, 'IDLH 100 ppm'), (10, '노출기준 TWA 10 ppm')):
        ax[1].axhline(v, color='k', lw=0.6, ls=':'); ax[1].text(0.12, v * 1.15, t, fontsize=7)
    ax[0].set_ylabel('바이오가스 (vol %)'); ax[1].set_ylabel('H₂S (ppm)')
    for a, t in zip(ax, ('(a) 플룸 축선 바이오가스 농도', '(b) 플룸 축선 H₂S 농도')):
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('배출구로부터 거리 (m)'); a.set_title(t, loc='left'); a.legend(fontsize=6.8)
    ax[0].set_ylim(0.05, 100); ax[1].set_ylim(1, 3e4)
    save(fig, 'f_near')

# ---------------------------------------------------------------- 4. 확산모델 정합성
def f_check():
    x = np.logspace(0, 3, 300)
    fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.9), gridspec_kw=dict(wspace=0.5))
    for cls, u, ls in (('D', 5.0, '-'), ('F', 2.0, '--')):
        b = cq.sig(x, cls); g = dc.ext_gpm(cls, x); f = dc.ext_rural(cls, x, u)
        ax[0].plot(x, b[0], color=C[0], ls=ls, label=f'Briggs {cls}'); ax[1].plot(x, b[1], color=C[0], ls=ls, label=f'Briggs {cls}')
        if g is not None: ax[0].plot(x, g[0], color=C[1], ls=ls, label=f'ISC3-PG {cls}'); ax[1].plot(x, g[1], color=C[1], ls=ls, label=f'ISC3-PG {cls}')
        if f is not None: ax[0].plot(x, f[0], color=C[2], ls=ls, label=f'TNO rural {cls}'); ax[1].plot(x, f[1], color=C[2], ls=ls, label=f'TNO rural {cls}')
        for (c2, xr), (syr, szr) in dc.PG_REF.items():
            if c2 == cls: ax[0].plot(xr, syr, 'k*', ms=6); ax[1].plot(xr, szr, 'k*', ms=6)
        bb = cq.sig(np.maximum(x, 1e-3), cls); near = cq.plume_centreline(141 / 3600, u, cls, x)
        ax[2].plot(x, dc.gauss_axis(bb[0], bb[1], u) * 100, color=C[0], ls=ls, label=f'가우시안·Briggs {cls}')
        if g is not None: ax[2].plot(x, dc.gauss_axis(g[0], g[1], u) * 100, color=C[1], ls=ls, label=f'가우시안·ISC3 {cls}')
        ax[2].plot(x, near * 100, color=C[3], ls=ls, label=f'적분 플룸 {cls}')
    for a, t, yl in zip(ax, ('(a) σy', '(b) σz', '(c) 축선 농도, 141 m³/h'), ('σy (m)', 'σz (m)', '바이오가스 (vol %)')):
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('거리 (m)'); a.set_ylabel(yl); a.set_title(t, loc='left'); a.tick_params(labelsize=7)
    ax[0].axvspan(1, 100, color='#efefef', zorder=0); ax[1].axvspan(1, 100, color='#efefef', zorder=0)
    ax[0].legend(fontsize=5.6, loc='upper left'); ax[2].legend(fontsize=5.6, loc='lower left'); ax[2].axhline(cq.LFL * 100, color='k', lw=0.5, ls=':'); ax[2].set_ylim(1e-4, 100)
    save(fig, 'f_check')

# ---------------------------------------------------------------- 5. 원거리 H2S
def f_far():
    x = np.logspace(-0.3, 3.3, 400); h_s = 18.25
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw=dict(wspace=0.4))
    for i, (lab, cls, u) in enumerate((('D, 5 m/s', 'D', 5.0), ('D, 3 m/s', 'D', 3.0), ('F, 2 m/s', 'F', 2.0))):
        ax[0].plot(x, gf.gaussian(84.5 / 3600, u, cls, x, 18.25, h_s) * cq.yH2S * 1e6, color=C[i], label=lab)
        ax[1].plot(x, gf.gaussian(84.5 / 3600, u, cls, x, 1.5, h_s) * cq.yH2S * 1e6, color=C[i], label=lab)
    for a in ax:
        for lv, t in ((41, 'AEGL-2 41'), (0.75, 'AEGL-1 0.75'), (0.01, '악취 0.01')):
            a.axhline(lv, color='k', lw=0.5, ls=':'); a.text(0.6, lv * 1.2, t, fontsize=6.5, color='#333333')
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('풍하 거리 (m)'); a.set_ylabel('H₂S 10분 평균 (ppm)'); a.legend(fontsize=6.8)
    ax[0].set_ylim(1e-3, 3e3); ax[1].set_ylim(1e-4, 1e2)
    ax[0].set_title('(a) 배출구 높이 (18.25 m)', loc='left'); ax[1].set_title('(b) 지표 (1.5 m)', loc='left')
    save(fig, 'f_far')

# ---------------------------------------------------------------- 6. 케이스별 위험도
def f_ir():
    BR = [('IR_resp_demand', '점화 실패 (개폐 반복)'), ('IR_resp_stop', '후단 정지'), ('IR_resp_foam', '거품 차단'), ('IR_resp_block', '응축수·동결·차단밸브'),
          ('IR_resp_wv', '무게추 밸브 고착'), ('IR_resp_R5', '거품 오염 → 실 파손'), ('IR_resp_HW', '기계적 미작동 → 실 파손'), ('IR_rout', '정기작업 중 동시 노출')]
    BC = [C[1], C[0], C[2], C[4], C[3], C[5], C[6], C[7]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6)); x = np.arange(4); bottom = np.zeros(4)
    for (col, name), cc in zip(BR, BC):
        v = np.array([pick(rb, c, f, col) for c, f in ORDER]); ax.bar(x, v, 0.55, bottom=bottom, color=cc, label=name); bottom += v
    iw = [pick(rb, c, f, 'IR_worker') for c, f in ORDER]; ax.plot(x, iw, 'kv', ms=6, label='최대노출 개인 IR')
    for xi, t, w in zip(x, bottom, iw): ax.text(xi, t * 1.12, f'{t:.1e}', ha='center', fontsize=7); ax.text(xi + 0.32, w, f'{w:.1e}', fontsize=6.5, va='center')
    for v, t in ((1e-3, '작업자 수용불가 1×10⁻³ (HSE)'), (1e-6, '광범위 수용 1×10⁻⁶')):
        ax.axhline(v, color='k', lw=0.6, ls=':'); ax.text(-0.42, v * 0.6, t, fontsize=6.8, ha='left', bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85))
    ax.set_yscale('log'); ax.set_ylim(1e-7, 1e-1); ax.set_xticks(x); ax.set_xticklabels([CASEL[k] for k in ORDER], fontsize=7.5)
    ax.set_ylabel('연간 사망 위험 (PLL, /년)'); ax.legend(fontsize=6.3, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.16))
    save(fig, 'f_ir')

# ---------------------------------------------------------------- 7. 시간 분포
def f_prof():
    prof = pd.read_csv(os.path.join(OUTD, 'risk_profiles.csv'))
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw=dict(wspace=0.35))
    m = prof[prof.kind == 'month'].copy(); m['datetime'] = pd.to_datetime(m['datetime'])
    for i, (c, f) in enumerate(ORDER):
        s = m[(m.case == c) & np.isclose(m.f_on, f)].sort_values('datetime'); s = s[s.datetime < '2021-11-01']
        ax[0].plot(range(len(s)), s['IR_total'] * 1e3, color=C[i], marker='o', ms=2.5, label=CASEL[(c, f)].replace('\n', ', '))
        h = prof[(prof.kind == 'hour') & (prof.case == c) & np.isclose(prof.f_on, f)].sort_values('hour')
        ax[1].plot(h['hour'], h['IR_total'] * 1e3, color=C[i], marker='o', ms=2.5)
    ax[0].set_xticks(range(0, 12, 2)); ax[0].set_xticklabels(['20.11', '21.01', '21.03', '21.05', '21.07', '21.09'], fontsize=7)
    ax[0].set_ylabel('월 기여 (×10⁻³ /년)'); ax[0].set_title('(a) 월별', loc='left', pad=26); ax[0].legend(fontsize=6.3, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0))
    ax[1].set_xlabel('시각'); ax[1].set_ylabel('시간대 기여 (×10⁻³ /년)'); ax[1].set_xticks(range(0, 24, 4)); ax[1].set_title('(b) 시각별', loc='left')
    save(fig, 'f_prof')

# ---------------------------------------------------------------- 8. 설계 대안
def f_opt():
    fig, ax = plt.subplots(figsize=(6.4, 3.0)); x = np.arange(4); w = 0.2
    names = [('현행', None), ('A: 개방 직접 지시', 'direct indication'), ('B: 독립 2차 방출 경로', 'second relief'), ('A + B', 'both')]
    for j, (nm, key) in enumerate(names):
        v = [pick(rb, c, f, 'IR_worker') if key is None else float(ro[(ro.case == f'{c} + {key}') & np.isclose(ro.f_on, f)]['IR_worker'].iloc[0]) for c, f in ORDER]
        ax.bar(x + (j - 1.5) * w, v, w, color=[C[7], C[0], C[2], C[1]][j], label=nm)
    ax.axhline(1e-3, color='k', lw=0.6, ls=':'); ax.text(3.5, 1.15e-3, '1×10⁻³', fontsize=7, ha='right')
    ax.set_yscale('log'); ax.set_ylim(1e-4, 3e-2); ax.set_xticks(x); ax.set_xticklabels([CASEL[k] for k in ORDER], fontsize=7.5); ax.set_ylabel('최대노출 개인 IR (/년)')
    ax.legend(ncol=4, fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, 1.12))
    save(fig, 'f_opt')

# ---------------------------------------------------------------- 9. 토네이도
KOR = {'PVRV fouled by foam 0.03 / 0.3': '거품의 통기밸브 오염 0.03 / 0.3', 'seal-failure injury 0.02 / 0.3': '실 파손 시 치사 0.02 / 0.3',
       'fatality given flash in the small plume 0.03 / 1.0': '소형 플룸 화재 치사 0.03 / 1.0', 'manual vent fouled too 0.1 / 0.6': '수동 벤트 동시 오염 0.1 / 0.6',
       'foaming 1 / 3 per year': '거품 발생 1 / 3 회/년', 'prolonged stay in the H2S plume 0.005 / 0.3 (no detector)': 'H₂S 플룸 체류 0.005 / 0.3',
       'H2S in the gas 500 ppm (in-situ desulphurisation) / 5,845 ppm (max)': '가스 중 H₂S 500 / 5,845 ppm', 'weather: all D5 / all calm': '기상 전부 D5 / 전부 무풍',
       'duty cycle ignored (time-mean plume) / pulses at full stay': '펄스 무시 / 전 체류 개방', 'ignition probability 0.005 / 0.05': '점화확률 0.005 / 0.05',
       'time at the valve 2 / 15 min': '밸브 앞 체류 2 / 15분', 'flare PFD 0.01 / 0.10': '연소설비 PFD 0.01 / 0.10', 'hard-fault share 0 / 0.3': '고장 비율 0 / 0.3',
       'routine roof work 0.5 / 6 h per week': '정기작업 0.5 / 6 h/주', 'P(inside flammable plume at the valve) 0.03 / 0.3': '가연 플룸 안 위치 0.03 / 0.3',
       'worker ignition source 0 / 0.01': '작업자 점화원 0 / 0.01', 'near the valve during routine work 0.1 / 1.0': '정기작업 중 밸브 근접 0.1 / 1.0',
       'responder goes to the roof: all branches x0.5 / x2 (capped at 1)': '대응자 지붕 접근 ×0.5 / ×2', 'weather: KMA hourly (base) / former assumed day-night mix': '기상 KMA 시간별 / 주야 비율 가정',
       'PAH available (base) / no PAH: roof visits x1.5, plume stay x2 (foam, block)': 'PAH 있음 / 없음',
       'H2S probit: Bevi 2009 / RIVM 2018 (base)': 'H₂S 프로빗 Bevi 2009 / RIVM 2018', 'H2S content: in-situ desulphurisation x0.2 / x1.5': '가스 중 H₂S ×0.2 (탈황) / ×1.5',
       'H2S: annual mean 2,527 ppm / hourly ADM1 (base)': 'H₂S 연평균 / 시간별', 'near-field concentration: free plume x0.5 / recirculation x2': '근거리 농도 ×0.5 / ×2 (재순환)',
       'plume entrainment beta 0.8 / 0.4': '플룸 혼입계수 β 0.8 / 0.4', 'seal fails at 12 / 8 kPag': '실 파손 압력 12 / 8 kPag',
       'PVRV fail-to-open: clean (base) / dirty service x5': '기계적 미작동 ×1 / ×5 (오염 환경)', 'PVRV lambda_DU 1.1e-6 (base) / 2.2e-6 per h': 'λ_DU 1.1 / 2.2×10⁻⁶ /h',
       'common cause beta 0.05 / 0.2': '공통원인 β 0.05 / 0.2', 'CO2 toxicity: excluded / included (base)': 'CO₂ 독성 제외 / 포함',
       'CH4/CO2/H2O: annual mean / hourly ADM1 (base)': 'CH₄·CO₂·H₂O 연평균 / 시간별'}
def f_tornado():
    s = pd.read_csv(os.path.join(OUTD, 'risk_sensitivity.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 4.2), gridspec_kw=dict(wspace=1.15))
    for a, f in zip(axs, (1.0, 0.8)):
        d = s[(s.case == 1) & np.isclose(s.f_on, f)].copy(); d['span'] = (np.log10(d[['low', 'high']].max(axis=1)) - np.log10(d[['low', 'high']].min(axis=1)))
        d = d.sort_values('span').tail(12); b = d['base'].iloc[0]; y = np.arange(len(d))
        a.barh(y, np.log10(d['low']) - np.log10(b), left=np.log10(b), color=C[0], height=0.6, label='낮은 값')
        a.barh(y, np.log10(d['high']) - np.log10(b), left=np.log10(b), color=C[1], height=0.6, label='높은 값')
        a.axvline(np.log10(b), color=INK, lw=0.7); a.set_yticks(y); a.set_yticklabels([KOR.get(p, p) for p in d['parameter']], fontsize=6.5); a.grid(axis='y', visible=False)
        tv = [1e-3, 1e-2, 1e-1]; a.set_xticks(np.log10(tv)); a.set_xticklabels(['10⁻³', '10⁻²', '10⁻¹'], fontsize=6.8); a.set_xlim(-3.2, -0.2)
        a.set_xlabel(f'PLL (기준 {b:.1e})', fontsize=7.5); a.set_title(f'Case 1, 기동 {int(f * 100)} %', loc='left'); a.legend(fontsize=6.3, loc='lower right')
    save(fig, 'f_tornado')

# ---------------------------------------------------------------- 10. 베이지안 갱신
def f_upd():
    u = pd.read_csv(os.path.join(OUTD, 'updating.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw=dict(wspace=0.32))
    for a, f in zip(axs, (1.0, 0.8)):
        for i, tp in enumerate((0.02, 0.05, 0.10)):
            s = u[(u.case == 1) & np.isclose(u.f_on, f) & np.isclose(u.true_pfd, tp)].sort_values('year')
            a.plot(s['year'], s['IR'], color=C[i], marker='o', ms=3, label=f'실제 PFD {tp}, 거품 {int(s.true_foam.iloc[0])}회/년'); a.fill_between(s['year'], s['IR_lo'], s['IR_hi'], color=C[i], alpha=0.12)
        a.set_yscale('log'); a.set_xlabel('운전 기록 연수'); a.set_ylabel('PLL (/년)'); a.set_title(f'Case 1, 기동 {int(f * 100)} %', loc='left'); a.axhline(1e-3, color='k', lw=0.5, ls=':')
    axs[0].legend(fontsize=6.3, loc='lower left')
    save(fig, 'f_upd')

# ---------------------------------------------------------------- 11. 정적 대 동적, 영향별
def f_static():
    sd = pd.read_csv(os.path.join(OUTD, 'static_vs_dynamic.csv')); eb = pd.read_csv(os.path.join(OUTD, 'risk_by_effect.csv'))
    g = float(r[r.case == 'generic']['IR_total'].iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw=dict(wspace=0.38)); x = np.arange(4)
    for j, (col, nm, cc) in enumerate((('static', '정적 (연평균, 같은 시나리오)', C[0]), ('dynamic', '동적 (시간별)', C[1]))):
        v = [pick(sd, c, f, col) for c, f in ORDER]; ax[0].bar(x + (j - 0.5) * 0.36, v, 0.36, color=cc, label=nm)
    ax[0].axhline(g, color=C[7], ls='--', lw=0.8); ax[0].text(-0.45, g * 0.45, f'일반 빈도 QRA {g:.1e}', fontsize=6.5, ha='left', color='#444444', bbox=dict(fc='white', ec='none', pad=0.5))
    ax[0].set_yscale('log'); ax[0].set_ylim(3e-6, 1e-1); ax[0].set_xticks(x); ax[0].set_xticklabels([CASEL[k] for k in ORDER], fontsize=6.5); ax[0].set_ylabel('PLL (/년)')
    ax[0].set_title('(a) 일반 빈도·정적·동적', loc='left'); ax[0].legend(fontsize=6.2, loc='upper right')
    bottom = np.zeros(4)
    for col, nm, cc in (('PLL_h2so', 'H₂S 독성', C[4]), ('PLL_co2_increment', 'CO₂ 독성 추가분', C[5]), ('PLL_flash', '플래시 화재', C[3]), ('PLL_mech', '실 파손 (기계적)', C[7])):
        v = np.array([pick(rb, c, f, col) for c, f in ORDER]) * 1e3; ax[1].bar(x, v, 0.55, bottom=bottom, color=cc, label=nm); bottom += v
    ax[1].set_xticks(x); ax[1].set_xticklabels([CASEL[k] for k in ORDER], fontsize=6.5); ax[1].set_ylabel('PLL (×10⁻³ /년)'); ax[1].set_title('(b) 영향별 PLL', loc='left'); ax[1].legend(fontsize=6.2)
    save(fig, 'f_static')

if __name__ == '__main__':
    for fn in (f_frame, f_met, f_near, f_check, f_far, f_ir, f_prof, f_opt, f_tornado, f_upd, f_static): fn()
