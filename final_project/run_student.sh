#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=runmxnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000m 
#SBATCH --time=00:20:00
#SBATCH --partition=a100
#SBATCH --gpus-per-node=1
# "#SBATCH --nodelist=gpu03"

module purge
module load gcc/8.3.1
module load cudnn/8.2.4.15-11.4
module load cuda/11.4.0
module load intel-oneapi-mkl/2022.1.0

#ncu python2 submit/submission.py
python2 submit/submission.py
