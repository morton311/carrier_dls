#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=challenge
#SBATCH --account=NAWCP24632466
#SBATCH --qos=debug
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 1:00:00
#SBATCH --output=out/challenge.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

# torchrun main.py -c 'challenge' -m 'train' -d "True"
python main.py -c challenge -m pred
python main.py -c challenge -m eval
