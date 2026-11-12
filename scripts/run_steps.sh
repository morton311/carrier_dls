#!/usr/bin/env bash
# Run explicit main.py steps in order: run_steps.sh config:mode,mode config:mode ...
# Modes are main.py modes (train, pred, eval); 'retrain' is train with -o m (from scratch).
PY=/home/cmorton/miniconda3/envs/pytorch_env/bin/python
cd /home/HDD/cmorton/carrier_dls || exit 1
for item in "$@"; do
    c=${item%%:*}; modes=${item#*:}
    for m in ${modes//,/ }; do
        args=(-m "$m"); [ "$m" = retrain ] && args=(-m train -o m)
        "$PY" main.py -c "$c" "${args[@]}" > /dev/null 2>&1 || { echo "FAILED $c at $m $(date '+%m-%d %H:%M')"; continue 2; }
    done
    echo "EXIT $c $(date '+%m-%d %H:%M')"
done
