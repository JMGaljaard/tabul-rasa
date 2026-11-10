#!/bin/zsh
source /opt/ohpc/admin/lmod/8.5.22/init/profile
module load cuda12.3/toolkit/12.3

SINGULARITYENV_CUDA_HOME=$CUDA_HOME apptainer build --build-arg SSH_AUTH_SOCK=$SSH_AUTH_SOCK  --force --bind /cm:/cm --nv  --fakeroot /var/scratch/$USER/tabdiff-deps.sif slurm/Apptainer.deps.def 

