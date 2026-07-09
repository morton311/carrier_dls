#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=pod_ldc
#SBATCH --account=AFMNG31652E99
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH --output=out/pod_ldc.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

torchrun main.py -c ldc/pod -m train -d True 
python main.py -c ldc/pod -m pred
python main.py -c ldc/pod -m eval