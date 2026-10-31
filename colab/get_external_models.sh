#!/usr/bin/env bash
# Clone the two independent Gaussian codes used as cross-checks by dynamic_qra/gaussian_farfield.py and
# dynamic_qra/dispersion_check.py. They are not vendored, because neither states a licence:
#   GPM            - EPA ISC3 fits of the Pasquill-Gifford curves (P. Connolly's teaching code, University of Manchester, 2017)
#   gaussian-plume - TNO rural sigma scheme (package option mode='farm'), EU LIFE CLINSH
set -e
cd "$(dirname "$0")/.."
mkdir -p external
[ -d external/GPM ] || git clone --depth 1 https://github.com/koaandmud/GPM external/GPM
[ -d external/gaussian-plume ] || git clone --depth 1 https://github.com/gerritjandebruin/gaussian-plume external/gaussian-plume
echo "external models ready: external/GPM, external/gaussian-plume"
