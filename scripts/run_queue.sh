#!/usr/bin/env bash
# Train, forecast and evaluate configs in order, e.g.
#   scripts/run_queue.sh challenge/a challenge/b
# Prints "EXIT <config>" after each one; a failed step skips the rest of that config.
PY=/home/cmorton/miniconda3/envs/pytorch_env/bin/python
cd "$(dirname "$0")/.." || exit 1
for c in "$@"; do
    for m in train pred eval; do
        "$PY" main.py -c "$c" -m "$m" > /dev/null 2>&1 || { echo "FAILED $c at $m"; break; }
    done
    echo "EXIT $c $(date '+%m-%d %H:%M')"
done
