#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=carrier
#SBATCH --account=AFMNG31652E99
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH --output=out/carrier.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

torchrun main.py -c 'carrier/multi_train' -m 'train' -d "True"
