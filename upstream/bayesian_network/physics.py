# -*- coding: utf-8 -*-
"""
Physical response functions used by the BN. Calibrated against the 1-s model (pressure_model/pressure_model.py)
and the downstream-stop event scan (pressure_model/scan_stop.py).

  * holder free-membrane stiffness   k_free = 0.0706 kPa per reference m3
  * holder full (rigid) stiffness    k_full = 0.0803 kPa per reference m3   (headspaces 800 + piping 1.6 + holder 500)
  * one digester isolated            k_iso  = 0.266  kPa per reference m3   (headspace 400)
  * effective margin  margin_eff = 5.05 (opening pressure incl. 1 % stiction) - p_op - 0.11
    (elevation head 0.03 + friction 0.02-0.05 + weight stiction 0.04, roughly an upper bound)
      -> matches the event-scan medians (holder full -> first opening 8.6 / 17.5 / 26.5 min at Q = 82 m3/h)
"""
import numpy as np, pandas as pd

P_REF = 101.325
K_FREE, K_FULL, K_ISO = 0.0706, 0.0803, 0.266
V_MAX = 500.0
P_OP = {1: 4.0, 2: 3.0, 3: 2.0}
P_OPEN = 5.05                      # setting 5.0 + stiction 1 %
OFFSET = 0.11                      # elevation head + friction + weight stiction (kPa)

def margin_eff(case):
    return P_OPEN - P_OP[case] - OFFSET

def t_full_h(fill, q_net, p_atm=101.6, case=1):
    """Time (h) for the holder to fill up from fill fraction `fill`. q_net: net inflow to the holder (m3/h, reference state)"""
    cap = (1.0 - fill) * V_MAX * (p_atm + P_OP[case]) / P_REF
    q = np.asarray(q_net, float)
    return np.where(q > 0, cap / np.maximum(q, 1e-9), np.inf)

def t_rigid_h(case, q_gen):
    """Time (h) from holder full to PVRV opening"""
    return margin_eff(case) / (K_FULL * np.maximum(np.asarray(q_gen, float), 1e-9))

def t_iso_h(case, q_one):
    """Time (h) to opening after the outlet of one digester is blocked. q_one: gas production of that digester"""
    return margin_eff(case) / (K_ISO * np.maximum(np.asarray(q_one, float), 1e-9))

def wv_stuck_ratchet(patm_h, case, latent_h=168, dp_bl=0.3):
    """Time (h) at which the accumulated atmospheric-pressure drop reaches the PVRV after the vent valve sticks closed.
    The blower keeps working (returns the air side from p_op - 0.3 to p_op).
    Returns: time to opening per start hour (nan if not reached within the latent period), and the gas vented through the PVRV
    from the opening until the end of the latent period (reference m3)."""
    p = np.asarray(patm_h, float)
    n = len(p) - latent_h
    g = np.full(n, P_OP[case]); t_open = np.full(n, np.nan); vent = np.zeros(n)
    m = margin_eff(case) + OFFSET     # holder gauge basis (the PVRV opens at holder gauge + OFFSET)
    for k in range(1, latent_h + 1):
        dp = p[k:k + n] - p[k - 1:k - 1 + n]          # atmospheric pressure change in this hour
        g = g - dp
        opened = ~np.isnan(t_open)
        # blower: when the gauge falls below p_op - 0.3 it is restored to p_op
        low = g < P_OP[case] - dp_bl
        g = np.where(low, P_OP[case], g)
        # PVRV: opens when the holder gauge reaches p_op + m (held there afterwards, the excess is vented)
        hit = g >= P_OP[case] + m
        newly = hit & ~opened
        t_open[newly] = k
        excess = np.where(hit, g - (P_OP[case] + m), 0.0)
        vent += excess / K_FREE
        g = np.where(hit, P_OP[case] + m, g)
    return t_open, vent

def operating_year(qh, use=60.0, Qf=150.0, fmin=0.25, on=0.85, off=0.70, L_derate=0.30, L_stop=0.15, L_resume=0.40, dt=60.0, fill0=0.5):
    """One-year operation with constant consumption + fill-level-linked flare. Returns 1-min fill fractions and the flare start times (s)"""
    t = np.arange(0, (len(qh) - 1) * 3600, dt); q = np.interp(t / 3600, np.arange(len(qh)), qh)
    V = fill0 * V_MAX; fl = False; use_on = True
    fills = np.empty(len(t)); starts = []
    st = dict(flared_m3=0.0, used_m3=0.0, aux_m3=0.0, derate_h=0.0, stop_h=0.0, flare_h=0.0)
    span = max(on + 0.10, off + 0.25) - off            # proportional band: 0 at off, rated at off + span
    for i, qi in enumerate(q):
        f = V / V_MAX
        if use_on and f <= L_stop: use_on = False
        if (not use_on) and f >= L_resume: use_on = True
        u = 0.0 if not use_on else (min(use, 0.9 * qi) if f < L_derate else use)
        if not fl and f >= on: fl = True; starts.append(t[i])
        if fl and f <= off: fl = False
        qf = 0.0 if not fl else min(Qf, max(fmin * Qf, Qf * (f - off) / span))
        V = min(max(V + (qi - u - qf) * dt / 3600, 0.0), V_MAX)
        fills[i] = V / V_MAX
        st['flared_m3'] += qf * dt / 3600; st['used_m3'] += u * dt / 3600; st['aux_m3'] += (use - u) * dt / 3600
        st['derate_h'] += dt / 3600 if (use_on and u < use) else 0.0; st['stop_h'] += dt / 3600 if not use_on else 0.0
        st['flare_h'] += dt / 3600 if qf > 0 else 0.0
    st['flare_starts'] = len(starts); st['production_m3'] = float(np.sum(q) * dt / 3600)
    return t, fills, np.array(starts), st
