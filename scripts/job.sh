#!/bin/bash
python -u main.py -c ldc/gram_perdof -m train
python -u main.py -c ldc/gram_perdof -m pred
python -u main.py -c ldc/gram_perdof -m eval