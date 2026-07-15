#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=challenge_global
#SBATCH --account=NAWCP24632466
#SBATCH --qos=debug
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 00:30:00
#SBATCH --output=out/challenge_global.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

# torchrun main.py -c challenge/global2 -m train -d True
python main.py -c challenge/global2 -m pred
python main.py -c challenge/global2 -m eval
