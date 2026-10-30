# -*- coding: utf-8 -*-
"""
Stage 0b: hourly meteorology and Pasquill-Gifford stability class from the KMA ASOS 138 (Pohang) record.

Input  : data/kma_asos138_pohang_hourly_met.csv  (converted from the KMA Open MET Data Portal download by --convert <raw.csv>)
Output : out/met_hourly.csv  one row per hour of the study year, used by risk.py
         out/met_summary.csv, fig/met_stability.png

Method
  * Solar elevation at the middle of the hour (NOAA solar-position equations), Pohang 36.0326 N, 129.3797 E, KST = UTC+9.
  * Turner (1964) net-radiation-index method as used in the EPA STAR / ISC meteorological pre-processors:
      day   : insolation class from solar elevation (>60: 4, 35-60: 3, 15-35: 2, <15: 1), corrected for cloud:
              total cloud <= 5/10 no change; > 5/10 and ceiling < 7,000 ft: -2; 7,000-16,000 ft: -1; >= 16,000 ft and 10/10: -1
              (a corrected index below 1 is set to 1); 10/10 with ceiling < 7,000 ft gives 0 (neutral)
      night : 10/10 with ceiling < 7,000 ft gives 0; total cloud > 4/10: -1; <= 4/10: -2
      class : Turner's table of net radiation index vs wind speed in knots; class G is merged with F
  * Cross-check for daytime: the EPA solar-radiation / wind-speed (SRDT) scheme (EPA-454/R-99-005, table 6-7: 925 / 675 / 175 W/m2)
    with the KMA hourly global radiation; agreement is reported.
  * Wind at the vent height (18.25 m) from the 10-m wind with the ISC rural power-law exponents (A-F: .07 .07 .10 .15 .35 .55).
  * Calm: 10-m wind below 0.5 m/s (the KMA record reports calms as 0.0); these hours go to the calm-air (MTT) plume.
  * Missing hours (the download ends at 2021-10-31 00:00, 24 h short of the study year) are filled with the same hour of the previous day
    and flagged.
"""
import os, sys, math, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
OUT = os.path.join(HERE, 'out'); FIG = os.path.join(HERE, 'fig'); os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
MET_CSV = os.path.join(ROOT, 'data', 'kma_asos138_pohang_hourly_met.csv')
LAT, LON = 36.0326, 129.3797
H_VENT, Z_REF = 18.25, 10.0
P_EXP = {'A': 0.07, 'B': 0.07, 'C': 0.10, 'D': 0.15, 'E': 0.35, 'F': 0.55}
FT = 0.3048
# Turner's table: rows = wind-speed bins (knots, upper bounds inclusive), columns = net radiation index 4, 3, 2, 1, 0, -1, -2
TURNER = [(1, [1, 1, 2, 3, 4, 6, 7]), (3, [1, 2, 2, 3, 4, 6, 7]), (5, [1, 2, 3, 4, 4, 5, 6]), (6, [2, 2, 3, 4, 4, 5, 6]),
          (7, [2, 2, 3, 4, 4, 4, 5]), (9, [2, 3, 3, 4, 4, 4, 5]), (10, [3, 3, 4, 4, 4, 4, 5]), (11, [3, 3, 4, 4, 4, 4, 4]), (1e9, [3, 4, 4, 4, 4, 4, 4])]
CLS = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'F'}
RAW_COLS = {'일시': 'datetime', '기온(°C)': 'T_C', '풍속(m/s)': 'u10_ms', '풍향(16방위)': 'wd_deg', '습도(%)': 'RH_pct', '현지기압(hPa)': 'p_hPa',
            '일조(hr)': 'sunshine_h', '일사(MJ/m2)': 'solar_MJm2', '전운량(10분위)': 'cloud_10', '중하층운량(10분위)': 'low_cloud_10', '최저운고(100m )': 'ceiling_100m'}

def convert(raw):
    """KMA download (EUC-KR / CP949, Korean headers) -> UTF-8 file with English headers"""
    for enc in ('utf-8-sig', 'cp949'):
        try:
            d = pd.read_csv(raw, encoding=enc); break
        except UnicodeDecodeError:
            continue
    d = d.rename(columns={k.strip(): v for k, v in RAW_COLS.items()}).rename(columns=lambda c: RAW_COLS.get(c, c))
    keep = [c for c in RAW_COLS.values() if c in d.columns]
    d[keep].to_csv(MET_CSV, index=False); print(f'written {MET_CSV}: {len(d)} rows, columns {keep}')

def solar_elevation(t_local):
    """NOAA solar position; t_local: DatetimeIndex in KST. Returns degrees."""
    t = t_local - pd.Timedelta(hours=9)
    jd = t.to_julian_date().to_numpy(); jc = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360; M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    C = np.sin(np.radians(M)) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + np.sin(np.radians(2 * M)) * (0.019993 - 0.000101 * jc) + np.sin(np.radians(3 * M)) * 0.000289
    lam = L0 + C - 0.00569 - 0.00478 * np.sin(np.radians(125.04 - 1934.136 * jc))
    eps = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60 + 0.00256 * np.cos(np.radians(125.04 - 1934.136 * jc))
    dec = np.degrees(np.arcsin(np.sin(np.radians(eps)) * np.sin(np.radians(lam))))
    y = np.tan(np.radians(eps / 2)) ** 2
    eqt = 4 * np.degrees(y * np.sin(2 * np.radians(L0)) - 2 * e * np.sin(np.radians(M)) + 4 * e * y * np.sin(np.radians(M)) * np.cos(2 * np.radians(L0))
                          - 0.5 * y * y * np.sin(4 * np.radians(L0)) - 1.25 * e * e * np.sin(2 * np.radians(M)))
    minutes = t.hour.to_numpy() * 60 + t.minute.to_numpy()
    tst = (minutes + eqt + 4 * LON) % 1440; ha = tst / 4 - 180
    cz = np.sin(np.radians(LAT)) * np.sin(np.radians(dec)) + np.cos(np.radians(LAT)) * np.cos(np.radians(dec)) * np.cos(np.radians(ha))
    return 90 - np.degrees(np.arccos(np.clip(cz, -1, 1)))

def turner_class(elev, cloud, ceil_m, u10):
    knots = u10 / 0.514444
    day = elev > 0
    ceil_ft = np.where(np.isnan(ceil_m), 1e9, ceil_m / FT)
    nri = np.zeros(len(elev), int)
    for i in range(len(elev)):
        c, cf = cloud[i], ceil_ft[i]
        if c >= 10 and cf < 7000:
            n = 0
        elif day[i]:
            e = elev[i]; n = 4 if e > 60 else 3 if e > 35 else 2 if e > 15 else 1
            if c > 5:
                if cf < 7000: n -= 2
                elif cf < 16000: n -= 1
                elif c >= 10: n -= 1
            n = max(n, 1)
        else:
            n = -1 if c > 4 else -2
        nri[i] = n
    col = {4: 0, 3: 1, 2: 2, 1: 3, 0: 4, -1: 5, -2: 6}
    out = []
    for k, n in zip(knots, nri):
        kk = round(k)
        for ub, row in TURNER:
            if kk <= ub:
                out.append(CLS[row[col[n]]]); break
    return np.array(out), nri

def srdt_day_class(rad_kw, u10):
    """daytime cross-check: EPA solar-radiation / wind-speed (SRDT) scheme, EPA-454/R-99-005 (2000) table 6-7:
    radiation >= 925, 675-925, 175-675, < 175 W/m2 x wind < 2, 2-3, 3-5, 5-6, >= 6 m/s"""
    rad = np.asarray(rad_kw) * 1000.0
    ri = np.where(rad >= 925, 0, np.where(rad >= 675, 1, np.where(rad >= 175, 2, 3)))
    tab = [['A', 'A', 'B', 'D'], ['A', 'B', 'C', 'D'], ['B', 'B', 'C', 'D'], ['C', 'C', 'D', 'D'], ['C', 'D', 'D', 'D']]   # rows: wind bins
    ub = np.digitize(u10, [2, 3, 5, 6])
    return np.array([tab[j][i] for i, j in zip(ri, ub)])

def build(year_index=None):
    m = pd.read_csv(MET_CSV, parse_dates=['datetime']).set_index('datetime')
    if year_index is None: year_index = pd.date_range('2020-11-01 01:00', '2021-11-01 00:00', freq='h')
    m = m[~m.index.duplicated()]
    filled = ~year_index.isin(m.index)
    m = m.reindex(year_index)
    for i in np.where(filled)[0]:                                                  # same hour of the previous day
        j = year_index.get_loc(year_index[i] - pd.Timedelta(days=1)); m.iloc[i] = m.iloc[j]
    m['filled'] = filled.astype(int)
    m['solar_elev_deg'] = solar_elevation(m.index - pd.Timedelta(minutes=30))
    ceil_m = m['ceiling_100m'].to_numpy(float) * 100
    cls, nri = turner_class(m['solar_elev_deg'].to_numpy(), m['cloud_10'].fillna(0).to_numpy(), ceil_m, m['u10_ms'].to_numpy())
    m['NRI'] = nri; m['PG_class'] = cls
    m['calm'] = (m['u10_ms'] < 0.5).astype(int)
    rad = m['solar_MJm2'].fillna(0).to_numpy() / 3.6                               # MJ/m2 per hour -> kW/m2
    dayc = srdt_day_class(rad, m['u10_ms'].to_numpy())
    m['PG_class_radiation'] = np.where(m['solar_elev_deg'] > 0, dayc, m['PG_class'])
    p = m['PG_class'].map(P_EXP).to_numpy()
    m['u_vent_ms'] = np.maximum(m['u10_ms'].to_numpy(), 0.5) * (H_VENT / Z_REF) ** p
    return m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--convert', help='raw KMA ASOS hourly CSV to convert into data/'); a = ap.parse_args()
    if a.convert: convert(a.convert)
    m = build(); m.to_csv(os.path.join(OUT, 'met_hourly.csv'), float_format='%.3f', index_label='datetime')
    day = m['solar_elev_deg'] > 0
    s = pd.DataFrame({'all': m['PG_class'].value_counts(normalize=True), 'day': m.loc[day, 'PG_class'].value_counts(normalize=True),
                      'night': m.loc[~day, 'PG_class'].value_counts(normalize=True)}).fillna(0).sort_index()
    s.loc['calm (<0.5 m/s)'] = [m['calm'].mean(), m.loc[day, 'calm'].mean(), m.loc[~day, 'calm'].mean()]
    s.loc['mean u10 (m/s)'] = [m['u10_ms'].mean(), m.loc[day, 'u10_ms'].mean(), m.loc[~day, 'u10_ms'].mean()]
    s.to_csv(os.path.join(OUT, 'met_summary.csv'), float_format='%.4f'); print(s.round(3).to_string())
    agree = (m.loc[day, 'PG_class'] == m.loc[day, 'PG_class_radiation']).mean()
    adj = (abs(m.loc[day, 'PG_class'].map(ord) - m.loc[day, 'PG_class_radiation'].map(ord)) <= 1).mean()
    print(f'daytime agreement Turner vs radiation/wind scheme: exact {agree:.2f}, within one class {adj:.2f}; filled hours {int(m.filled.sum())}')
    # figure
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'axes.titleweight': 'bold'})
    col = {'A': '#d62728', 'B': '#ff7f0e', 'C': '#bcbd22', 'D': '#7f7f7f', 'E': '#1f77b4', 'F': '#17becf'}
    fig = plt.figure(figsize=(14, 4.2)); gs = fig.add_gridspec(1, 3, wspace=0.3)
    ax = fig.add_subplot(gs[0]); hr = m.groupby(m.index.hour)['PG_class'].value_counts(normalize=True).unstack().fillna(0)
    bottom = np.zeros(24)
    for c in 'ABCDEF':
        if c in hr: ax.bar(hr.index, hr[c], bottom=bottom, color=col[c], label=c, width=0.9); bottom += hr[c].to_numpy()
    ax.set_xlabel('hour of day (KST)'); ax.set_ylabel('share of hours'); ax.set_title('(a) stability class by hour', loc='left'); ax.legend(ncol=6, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    ax = fig.add_subplot(gs[1]); per = m.index.to_period('M'); full = per.value_counts(); mm = m[per.isin(full[full >= 24 * 25].index)]
    mo = mm.groupby(mm.index.to_period('M'))['PG_class'].value_counts(normalize=True).unstack().fillna(0); bottom = np.zeros(len(mo))
    for c in 'ABCDEF':
        if c in mo: ax.bar(range(len(mo)), mo[c], bottom=bottom, color=col[c], width=0.9); bottom += mo[c].to_numpy()
    ax.set_xticks(range(len(mo))); ax.set_xticklabels([p.strftime('%y-%m') for p in mo.index], rotation=90, fontsize=7); ax.set_title('(b) by month', loc='left')
    ax = fig.add_subplot(gs[2], projection='polar'); ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    bins = np.arange(0, 361, 22.5); wd = m.loc[m.calm == 0, 'wd_deg'].to_numpy() % 360
    sp = m.loc[m.calm == 0, 'u10_ms'].to_numpy(); base = np.zeros(16)
    for lo, hi, cc in ((0.5, 2, '#9ecae1'), (2, 4, '#4292c6'), (4, 99, '#08306b')):
        h, _ = np.histogram((wd[(sp >= lo) & (sp < hi)] + 11.25) % 360, bins=bins); h = h / len(m)
        ax.bar(np.radians(bins[:-1]), h, width=np.radians(22.5), bottom=base, color=cc, edgecolor='white', lw=0.3, label=f'{lo}-{hi if hi < 99 else ""} m/s'); base += h
    ax.set_title('(c) wind rose, 10 m (direction from)', loc='left'); ax.legend(fontsize=6.5, loc='lower left', bbox_to_anchor=(-0.25, -0.15)); ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(FIG, 'met_stability.png'), dpi=150, bbox_inches='tight'); plt.close(fig); print('fig ok')

if __name__ == '__main__':
    main()
