#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=ldc
#SBATCH --account=NAWCP24632466
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH --output=output/out_ldc.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

torchrun main.py -c 'ldc' -m 'train' -d "True"
