#!/usr/bin/env bash
set -e

STACK="/data/a.saricaoglu/repo/RubinsCSSL/stack"

export LSST_CONDA_ENV_NAME="rubin-cssl"
export PYTHONNOUSERSITE=1

source "$STACK/loadLSST.bash"
setup lsst_distrib

exec python -m ipykernel_launcher -f "$1"
