#!/usr/bin/env bash
# Reproduces every number and chart in the article (~20 min on a laptop CPU).
set -e
bash download_data.sh
python audit.py                                   # Part A: real duplicate audit
for f in 0 0.1 0.25 0.5 1.0; do python leak_run.py 64 $f 1 20; done
for f in 0.25 0.5 1.0; do python leak_run.py 64 $f 5 20; done
for a in "0 1" "0.5 1" "1.0 1" "1.0 5"; do python leak_run.py 512,256 $a 20; done
python make_charts.py
