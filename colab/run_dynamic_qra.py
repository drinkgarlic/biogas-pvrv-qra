# -*- coding: utf-8 -*-
"""
One-command driver for biogas-pvrv-qra.

    python colab/run_dynamic_qra.py              # source terms -> consequence -> far-field Gaussian -> event tree / risk -> figures   (~2 min)
    bash colab/get_external_models.sh            # optional: clone the independent Gaussian codes used in dispersion_check.py
    python colab/run_dynamic_qra.py --updating   # also the Bayesian-updating demonstration (~6 min)

Needs only this repository: upstream/ holds a pinned copy of the BN, its operating-model inputs, the two data files and the
pressure-model tables the source terms need (from github.com/drinkgarlic/biogas-pvrv-BN).
"""
import argparse, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); QRA = os.path.join(ROOT, 'dynamic_qra'); PY = sys.executable
def run(script):
    t = time.time(); print(f'>> {script}', flush=True)
    r = subprocess.run([PY, script], cwd=QRA, text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:]); raise SystemExit(f'failed: {script}')
    print('\n'.join(r.stdout.strip().split('\n')[-8:])); print(f'   done in {time.time() - t:.0f} s', flush=True)
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--updating', action='store_true', help='also run the Bayesian-updating demonstration')
    a = ap.parse_args()
    for s in ('met_stability.py', 'source_terms.py', 'consequence.py', 'gaussian_farfield.py', 'dispersion_check.py', 'jet_fire_check.py', 'risk.py'): run(s)
    if a.updating: run('updating.py')
    run('fig_qra.py')
    print('finished. Results: dynamic_qra/out, figures: dynamic_qra/fig')
if __name__ == '__main__':
    main()
