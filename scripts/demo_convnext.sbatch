#!/bin/bash

#SBATCH --job-name=Demo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --ntasks-per-node=1
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=./sbatch_log.out
#SBATCH --error=./sbatch_log.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=10
#SBATCH --account=owj@v100
####SBATCH --account=vby@v100
#SBATCH --hint=nomultithread

#PYTHON=/gpfswork/rech/mjp/uzj43um/conda-envs/audio_retrieval/bin/python
PYTHON=/gpfsstore/rech/djl/uzj43um/conda-envs/test_audio_convnext_inf/bin/python

# module purge
# module load pytorch-gpu/py3/1.11.0

# echo des commandes
set -x

date

BASEDIR=/gpfswork/rech/djl/uzj43um/audioset-convnext-inf
SCRIPT=$BASEDIR/demo_convnext_v2.py

srun $PYTHON -u $SCRIPT
