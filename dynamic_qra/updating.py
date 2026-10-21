# -*- coding: utf-8 -*-
"""
Dynamic QRA, stage 4: Bayesian updating of the two priors that drive the result as operating records accumulate.

  * flare PFD        Beta prior with mean 0.05 (a = 1, b = 19); each year brings ~2,000 flare demands and the observed failures
  * foaming rate     Gamma prior with mean 2 /yr (shape 2, rate 1); each year brings the observed foam events
Three synthetic record histories ("true" values 0.02 / 0.05 / 0.10 for the PFD, 1 / 2 / 3 per year for foaming) are drawn with
a fixed seed, the posteriors are updated year by year, and the risk model (risk.run_case) is re-evaluated with the posterior
means. The 5-95 % credible band of the IR is obtained by re-evaluating at the posterior quantiles (the IR is monotone in both).
Output: out/updating.csv, used by fig_qra.py.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import beta, gamma
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import risk as rk
OUT = os.path.join(HERE, 'out')
rng = np.random.default_rng(2026)
N_DEM = 2000; YEARS = 5
rows = []
for case, f_on in ((1, 1.0), (1, 0.8)):
    for true_pfd, true_foam in ((0.02, 1.0), (0.05, 2.0), (0.10, 3.0)):
        a, b = 1.0, 19.0; k, th = 2.0, 1.0            # Beta(a, b) for PFD, Gamma(shape k, rate th) for foaming
        for yr in range(0, YEARS + 1):
            if yr > 0:
                nf = rng.binomial(N_DEM, true_pfd); a += nf; b += N_DEM - nf
                nfoam = rng.poisson(true_foam); k += nfoam; th += 1.0
            pfd_mean, pfd_lo, pfd_hi = a / (a + b), beta.ppf(0.05, a, b), beta.ppf(0.95, a, b)
            lam_mean, lam_lo, lam_hi = k / th, gamma.ppf(0.05, k, scale=1 / th), gamma.ppf(0.95, k, scale=1 / th)
            ir = {}
            for tag, pf, lm in (('mean', pfd_mean, lam_mean), ('lo', pfd_lo, lam_lo), ('hi', pfd_hi, lam_hi)):
                out, _ = rk.run_case(case, f_on, prm_over={'pfd_flare': float(pf), 'lam_foam': float(lm)})
                s = rk.summarize(out, case, f_on); ir[tag] = (s['IR_total'], s['openings'])
            rows.append(dict(case=case, f_on=f_on, true_pfd=true_pfd, true_foam=true_foam, year=yr, pfd_mean=pfd_mean, pfd_lo=pfd_lo, pfd_hi=pfd_hi,
                             foam_mean=lam_mean, foam_lo=lam_lo, foam_hi=lam_hi, IR=ir['mean'][0], IR_lo=ir['lo'][0], IR_hi=ir['hi'][0], openings=ir['mean'][1]))
            print(f'case {case} f_on {f_on} true pfd {true_pfd} year {yr}: pfd {pfd_mean:.4f} [{pfd_lo:.4f}, {pfd_hi:.4f}]  foam {lam_mean:.2f}  IR {ir["mean"][0]:.2e} [{ir["lo"][0]:.2e}, {ir["hi"][0]:.2e}]', flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'updating.csv'), index=False, float_format='%.5g')
