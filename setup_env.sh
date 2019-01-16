#!/bin/bash
# for the moment, need to use the most recent nightly build
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev3python3/latest/x86_64-centos7-gcc62-opt/setup.sh
export PYTHONPATH="$PYTHONPATH:/home/zgubic/hmumu/fair-hmumu/lib/python3.6/site-packages"
export PYTHONPATH="$PYTHONPATH:/home/zgubic/hmumu/fair-hmumu"

export DATA="/data/atlassmallfiles/users/zgubic/hmumu/tf_ready"
