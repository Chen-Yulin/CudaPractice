#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=makemxnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4000m
#SBATCH --time=00:20:00
#SBATCH --partition=debuga100
#SBATCH --gres=gpu:1

module purge
#export PATH=$(echo $PATH | tr ':' '\n' | grep -v "/dssg/opt/icelake/linux-centos8-icelake/intel-2021.6.0/miniconda3-4.10.3-dxdfpbtotz7dztpivh27il3hl7rczlcl/condabin" | tr '\n' ':')
#export PATH=$(echo $PATH | tr ':' '\n' | grep -v "/dssg/opt/icelake/linux-centos8-icelake/intel-2021.6.0/miniconda3-4.10.3-dxdfpbtotz7dztpivh27il3hl7rczlcl/bin" | tr '\n' ':')
module load gcc/8.3.1
module load cudnn/8.2.4.15-11.4
module load cuda/11.4.0
module load intel-oneapi-mkl/2022.1.0

cp -fv new-forward.cuh incubator-mxnet/src/operator/custom
make -C incubator-mxnet USE_CUDA=1 USE_CUDA_PATH=/dssg/opt/icelake/linux-centos8-icelake/gcc-11.2.0/cuda-11.4.0-wlktjuelwnglhi5gircc36obzl6irzcp USE_CUDNN=0 USE_MKLDNN=0 USE_BLAS=mkl USE_INTEL_PATH=/dssg/opt/icelake/linux-centos8-icelake/gcc-11.2.0/intel-oneapi-mkl-2022.1.0-y7l73hfw3kxkt3qymuuqkup3ucdxqwcw/mkl/2022.1.0
pip install --user -e incubator-mxnet/python

echo "Finished building mxnet"
