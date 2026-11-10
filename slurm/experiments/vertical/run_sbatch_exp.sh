DATASET=${DATASET:-"adult"}
CONFIG_NAME=${CONFIG_NAME:-"tabdiff_unlearn_sweep_orth"}
echo "RUN ${DATASET} with config: ${CONFIG_NAME}"


sbatch --export ALL,CONFIG_NAME=$CONFIG_NAME,DATASET=$DATASET slurm/run_unlearning_tabdiff.sh

