#!/bin/zsh
# Two-lane dispatcher: launches ids listed in experiments/<slug>/QUEUE (one per line,
# optional "<id> <threads>") in order, keeping at most $LANES experiments running.
# An id is only launched once its predecessor id exists in the ledger, so ids line up.
cd "$(dirname "$0")/.."
SLUG=playground-series-s6e10; Q=experiments/$SLUG/QUEUE; LANES=${LANES:-2}
while true; do
  running=$(sqlite3 ledger.db "select count(*) from experiments where comp_slug='$SLUG' and status in ('queued','running')")
  next=""
  while read -r id thr; do
    [[ -z "$id" ]] && continue
    sqlite3 ledger.db "select 1 from experiments where exp_id='$id'" | grep -q 1 && continue
    next=$id; nthr=${thr:-5}; break
  done < $Q
  if [[ -n "$next" && $running -lt $LANES ]]; then
    prev=$(printf "e%04d" $(( 10#${next#e} - 1 )))
    if sqlite3 ledger.db "select 1 from experiments where exp_id='$prev'" | grep -q 1; then
      scripts/s6e10_launch.sh $next $nthr
      echo "$(date -u +%H:%M) dispatched $next"
    fi
  fi
  sleep 10
done
