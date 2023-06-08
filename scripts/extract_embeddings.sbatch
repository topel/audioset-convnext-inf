#!/bin/bash

#SBATCH --job-name=Embed
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --ntasks-per-node=1
#####SBATCH --qos=qos_gpu-t3
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=./sbatch_log.out
#SBATCH --error=./sbatch_log.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=10
#SBATCH --account=owj@v100
####SBATCH --account=vby@v100
#SBATCH --hint=nomultithread

PYTHON=/gpfswork/rech/mjp/uzj43um/conda-envs/audio_retrieval/bin/python

# module purge
# module load pytorch-gpu/py3/1.11.0

# echo des commandes
set -x

date

WORKSPACE_SCRATCH=/gpfsscratch/rech/djl/uzj43um/audioset_tagging   
DATASPACE=/gpfsstore/rech/djl/uzj43um/audioset

BASEDIR=/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset-convnext-inf
SCRIPT=$BASEDIR/pytorch/extract_embeddings.py

# export TORCH_DISTRIBUTED_DEBUG=INFO

srun $PYTHON -u $SCRIPT
