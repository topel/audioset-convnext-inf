#!/bin/bash

#SBATCH --job-name=Eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --ntasks-per-node=1
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=./eval_sbatch_log.out
#SBATCH --error=./eval_sbatch_log.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=10
#SBATCH --account=djl@v100
#SBATCH --hint=nomultithread

PYTHON=/gpfswork/rech/mjp/uzj43um/conda-envs/audio_retrieval/bin/python

# module purge
# module load pytorch-gpu/py3/1.11.0

# echo des commandes
set -x

date

# DATASET_DIR=/gpfsdswork/dataset/AudioSet   # Default first argument.
WORKSPACE_SCRATCH=/gpfsscratch/rech/djl/uzj43um/audioset_tagging   # Default second argument.
DATASPACE=/gpfsstore/rech/djl/uzj43um/audioset

BASEDIR=/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset_tagging_cnn
SCRIPT=$BASEDIR/pytorch/evaluate_convnext_on_audioset.py

# export TORCH_DISTRIBUTED_DEBUG=INFO

srun $PYTHON -u $SCRIPT \
     --tiny_path=/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset_tagging_cnn/pretrained_models/my_models/convnext_tiny_471mAP.pth \
     --h5_indexes_dir_path=/gpfsstore/rech/djl/uzj43um/audioset/hdf5s/indexes
