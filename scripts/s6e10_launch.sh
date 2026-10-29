#!/bin/zsh
# usage: scripts/s6e10_launch.sh <exp-id> <threads> [--final]
# Runs one experiment in the background, logging to experiments/<slug>/<id>/run.out.
cd "$(dirname "$0")/.."
exp=$1; thr=$2; shift 2
S6E10_THREADS=$thr nohup uv run medalist run playground-series-s6e10 --exp $exp "$@" \
  > experiments/playground-series-s6e10/$exp/run.out 2>&1 &
# wait until the ledger has queued this id so concurrent launches get the right ids
for i in {1..60}; do
  sqlite3 ledger.db "select 1 from experiments where exp_id='$exp'" | grep -q 1 && break
  sleep 1
done
echo "launched $exp ($thr threads)"
