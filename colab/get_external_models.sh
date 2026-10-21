#!/usr/bin/env bash
# Clone the independent Gaussian plume / puff implementations used as cross-checks by dynamic_qra/gaussian_farfield.py and
# dynamic_qra/dispersion_check.py. They are not vendored (GPM and gaussian-plume state no licence; the two MIT ones are kept
# external for symmetry).
set -e
cd "$(dirname "$0")/.."
mkdir -p external
[ -d external/GPM ] || git clone --depth 1 https://github.com/koaandmud/GPM external/GPM
[ -d external/gaussian-plume ] || git clone --depth 1 https://github.com/gerritjandebruin/gaussian-plume external/gaussian-plume
[ -d external/CH4_dispersion ] || git clone --depth 1 https://github.com/yunhal/CH4_dispersion external/CH4_dispersion
[ -d external/FastGaussianPuff ] || git clone --depth 1 https://github.com/Hammerling-Research-Group/FastGaussianPuff external/FastGaussianPuff
echo "external models ready: external/GPM, external/gaussian-plume, external/CH4_dispersion, external/FastGaussianPuff"
