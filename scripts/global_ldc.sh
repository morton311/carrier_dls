#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=global_ldc
#SBATCH --account=NAWCP24632466
#SBATCH --qos=debug
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 1:00:00
#SBATCH --output=out/global_ldc.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

torchrun main.py -c ldc/global -m train -d True
python main.py -c ldc/global -m pred
python main.py -c ldc/global -m eval
