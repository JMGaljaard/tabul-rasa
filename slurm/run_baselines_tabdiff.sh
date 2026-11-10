#!/bin/sh
## Set max runtime (24h)
#SBATCH --time=104:00:00
## Set parallelism (4)
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1


## Set NVidia A4000 (16GB) GPU
#SBATCH -C A4000
## Make CUDA device visible to host
#SBATCH --gres=gpu:1

## Host envirionmental variables
CUDA_VERSION="12.3"

## Singularity setup variables
## Default working directory of dis/pytorch-base:2.2.1+
WORK_DIR=/workspace/
IMAGE_NAME="tabdiff-build.sif"
IMAGE_PATH=/var/scratch/$USER


# Mount  points:
## CUDA
export APPTAINERENV_LD_LIBRARY_PATH=/cm/local/apps/cuda-driver/libs/current/lib64
export APPTAINERENV_DRIVER_PATH=/cm/local/apps/cuda-driver/libs/current/lib64
export APPTAINERENV_LD_LIBRARY_PATH=/cm/local/apps/cuda-driver/libs/current/lib64
export APPTAINER_BIND="/cm:/cm"

# Hacky solution
# Provide BIND for models to contais
export APPTAINER_BIND="/var/scratch/$USER/tabdiff-dis/ckpt:/workspace/ckpt,${APPTAINER_BIND}"
export APPTAINER_BIND="/var/scratch/$USER/tabdiff-dis/result:/workspace/result,${APPTAINER_BIND}"
export APPTAINER_BIND="/var/scratch/$USER/tabdiff-dis/wandb:/workspace/wandb,${APPTAINER_BIND}"
export APPTAINER_BIND="/var/scratch/$USER/tabdiff-dis/data:/workspace/data,${APPTAINER_BIND}"
export APPTAINER_BIND="/var/scratch/$USER/tabdiff-dis/synthetic:/workspace/synthetic,${APPTAINER_BIND}"

export APPTAINER_BIND="/home/$USER/repo/tabdiff/configs:/workspace/configs,${APPTAINER_BIND}"
export APPTAINER_BIND="/home/$USER/repo/tabdiff/multirun:/workspace/multirun,${APPTAINER_BIND}"
export APPTAINER_BIND="/home/$USER/repo/tabdiff/outputs:/workspace/outputs,${APPTAINER_BIND}"

## Run
source /home/$USER/repo/tabdiff/.env
source /opt/ohpc/admin/lmod/lmod/init/profile

module load "cuda${CUDA_VERSION}/toolkit/${CUDA_VERSION}"

echo $DATASET

mkdir -p ${DATASET}/out
mkdir -p ${DATASET}/err

echo "Running jobs..."
srun --output="${DATASET}/out/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.%N.out" \
     --error="${DATASET}/err/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.%N.err" \
     --jobid $SLURM_JOBID \
     zsh -c "apptainer exec \
         --pwd /workspace \
         --nv \
         /var/scratch/$USER/tabdiff-build.sif \
         python3.10 \
         untrain_model.py \
         -m \
         --config-name ${CONFIG_NAME} "
