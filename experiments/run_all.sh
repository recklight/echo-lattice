#!/usr/bin/env bash
# The simulation experiments after freeze.json: E1t alone on one worker (its wall times are the
# timing result), then E1-E9 in parallel worker processes. Each experiment writes
# results/<name>.csv and results/<name>.meta.json; progress goes to results/run_all.log.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}  # the Python of the environment where echo-lattice is installed
WORKERS=${WORKERS:-14}
mkdir -p results
LOG=results/run_all.log
[ -f freeze.json ] || { echo "freeze.json missing" >> "$LOG"; exit 1; }
echo "start $(date -u +%FT%TZ), workers $WORKERS" >> "$LOG"
"$PY" harness.py --experiment E1t --workers 1 --out results >> "$LOG" 2>&1 || echo "E1t failed with $?" >> "$LOG"
for e in E1 E1b E2 E3 E4 E5 E6 E9; do
    "$PY" harness.py --experiment "$e" --workers "$WORKERS" --out results >> "$LOG" 2>&1 \
        || echo "$e failed with $?" >> "$LOG"
done
echo "end $(date -u +%FT%TZ)" >> "$LOG"
