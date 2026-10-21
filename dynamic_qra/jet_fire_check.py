# -*- coding: utf-8 -*-
"""
Is an ignited PVRV release a jet fire?  Screening of the flame regime, size and radiation of an ignited vent release.

The PVRV relieves at 5.05 kPag. Two velocity scales exist:
  * at the pallet seat gap the gas can reach u_seat = Cd sqrt(2 dp / rho) (~70-90 m/s), but the gap is a thin annulus and
    the jet is turned and decelerated inside the valve body and the weather hood;
  * at the DN200 outlet the mean velocity is u_out = Q / A (1-12 m/s for 141-1,410 m3/h).
The regime is judged with the flame Froude number of Delichatsios (1993),
    Fr_f = u f_s^1.5 / [ (rho_e/rho_a)^0.25 (dT_f/T_a g d)^0.5 ],   Fr_f < 1 buoyancy-controlled, Fr_f > 5 momentum-controlled (jet fire),
and, for the flame size, with Heskestad's buoyant-flame correlation  L = 0.235 Qdot^0.4 - 1.02 d  (kW, m), which is valid up to
Q* ~ 1e5; the momentum-jet length of Kalghatgi / Chamberlain-type jet-fire models is printed for comparison only.
Radiation: single point source at mid-flame, radiant fraction chi_r = 0.15 (methane, small buoyant flames; CO2 dilution lowers it),
wind tilt after the AGA / API 521 form cos(theta) = u*^-0.5 for u* > 1, u* = u / (g Qdot d / (rho_a cp T_a))^(1/3).
Harm: Eisenberg probit Pr = -14.9 + 2.56 ln(q^(4/3) t / 1e4), q in W/m2, t in s.
Output: out/jet_fire_check.csv
"""
import os, sys, math, json
import numpy as np, pandas as pd
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import consequence as cq
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
G = 9.80665; T_A = 288.15; RHO_A = cq.RHO_A; RHO_G = cq.RHO_G; CP_A = 1.005
D = 0.20; A = math.pi * D ** 2 / 4
yCH4 = cq.props['yCH4'] if hasattr(cq, 'props') else 0.64
LHV_CH4 = 35.8e3            # kJ/m3 at 0 C, 1 atm
LHV = yCH4 * LHV_CH4 * 273.15 / 308.15      # kJ per m3 of biogas at 35 C
M_G = 24.69; F_S = M_G / (M_G + yCH4 * 9.52 * 28.96)     # stoichiometric mixture fraction (mass)
DT_T = 5.5                  # flame temperature rise / ambient (T_ad ~ 1,850 K for 64 % CH4 / 30 % CO2)
CHI_R = 0.15
S_L = 0.25                  # laminar burning velocity of the stoichiometric biogas-air mixture (m/s), ~0.36 for CH4
DP_SET = 5050.0; CD = 0.6

def eisenberg(q_kw, t_s):
    dose = (q_kw * 1e3) ** (4 / 3) * t_s
    return float(norm.cdf(-14.9 + 2.56 * math.log(dose / 1e4) - 5)) if dose > 0 else 0.0

def main():
    pd.set_option('display.width', 250)
    q_adm = pd.read_csv(os.path.join(HERE, '..', 'upstream', 'data', 'ADM1_hourly_1to9_Q200_6000m3_ffOTS1.csv'))['Q_gas_wet_m3_d_20C'] / 24
    q_max_prod = float(q_adm.max())
    u_seat = CD * math.sqrt(2 * DP_SET / RHO_G)
    rows = []
    for name, Qh in (('R3 isolated digester, continuous (41 m3/h)', 41.2), ('R2 downstream stop (84.5 m3/h)', 84.5), ('valve at minimum stable opening (141 m3/h)', 141.0),
                     (f'maximum hourly production through one valve ({q_max_prod:.0f} m3/h)', q_max_prod), ('rated PVRV capacity (1,410 m3/h, never reached here)', 1410.0)):
        Q = Qh / 3600; u_out = Q / A
        Qdot = Q * LHV                                                    # kW
        frf_out = u_out * F_S ** 1.5 / ((RHO_G / RHO_A) ** 0.25 * math.sqrt(DT_T * G * D))
        # seat gap jet: annulus area = Q / u_seat, hydraulic diameter of a thin annulus = 2 x gap
        gap = Q / u_seat / (math.pi * D); d_h = 2 * gap
        frf_seat = u_seat * F_S ** 1.5 / ((RHO_G / RHO_A) ** 0.25 * math.sqrt(DT_T * G * d_h))
        L_hesk = 0.235 * Qdot ** 0.4 - 1.02 * D
        Qstar = Qdot / (RHO_A * CP_A * T_A * math.sqrt(G * D) * D ** 2)
        L_jet = 200 * d_h * math.sqrt(RHO_G / RHO_A) * 1.0              # momentum-jet flame length ~ 200 d_eff for CH4 (Kalghatgi), seat jet as if free
        r = dict(release=name, Q_m3h=Qh, Qdot_kW=Qdot, u_outlet_ms=u_out, u_seat_ms=u_seat, Fr_f_outlet=frf_out, Fr_f_seat_jet=frf_seat, Qstar=Qstar,
                 L_flame_buoyant_m=L_hesk, L_if_free_seat_jet_m=L_jet)
        for wn, u in (('calm 0.5', 0.5), ('D3', 3.0), ('D5', 5.0)):
            ustar = u / (G * Qdot * D / (RHO_A * CP_A * T_A)) ** (1 / 3)
            th = 0.0 if ustar <= 1 else math.degrees(math.acos(ustar ** -0.5)); r[f'tilt_{wn}_deg'] = th
        P = CHI_R * Qdot                                                  # radiated power, kW
        for lv in (37.5, 12.5, 4.7, 1.6):
            r[f'R_{lv}kW_m'] = math.sqrt(P / (4 * math.pi * lv))         # distance from the flame centre
        q1 = P / (4 * math.pi * max(1.0, L_hesk / 2) ** 2)                # 1 m beside the vent, ~L/2 from the flame centre
        r['q_at_1m_kW_m2'] = q1; r['P_fat_10s_at_1m'] = eisenberg(q1, 10); r['P_fat_30s_at_1m'] = eisenberg(q1, 30)
        r['u_outlet_over_SL'] = u_out / S_L
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'jet_fire_check.csv'), index=False, float_format='%.4g')
    print(df.T.to_string())
    print(f'\nLHV {LHV:.0f} kJ/m3 biogas at 35 C, f_s {F_S:.3f}, seat-gap velocity {u_seat:.0f} m/s, maximum hourly production {q_max_prod:.0f} m3/h')

if __name__ == '__main__':
    main()
