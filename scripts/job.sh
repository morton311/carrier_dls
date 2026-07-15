#!/bin/bash
python -u main.py -c ldc/global_dls_try7 -m train
python -u main.py -c ldc/global_dls_try7 -m pred
python -u main.py -c ldc/global_dls_try7 -m eval