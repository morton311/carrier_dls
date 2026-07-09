#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=local_ldc_dls
#SBATCH --account=NAWCP24632466
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH --output=out/local_ldc_dls.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

torchrun main.py -c ldc/local -m train -d True 
python main.py -c ldc/local -m pred
python main.py -c ldc/local -m eval