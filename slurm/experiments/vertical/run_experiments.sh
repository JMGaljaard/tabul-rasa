#!/usr/bin/env zsh
DATASET=${DATASET:-"adult"}
CONFIG_NAME=${CONFIG_NAME:-"tabdiff_unlearn_sweep_orth"}
echo "RUN ${DATASET} with config: ${CONFIG_NAME}"
# Default is adult

# Gradient Ascent
declare -A dataset_lookup
 dataset_lookup[magic]="\"'1','2','3'\""        # Width, size, conc
dataset_lookup[adult]="\"'8','9','0'\""        # Width, size, conc
dataset_lookup[shoppers]="\"'8','7','10','15'\""        # Width, size, conc
dataset_lookup[default]="\"'5','1','6','7'\""        # Width, size, conc
dataset_lookup[beijing]="\"'5','4','7','1'\""        # Width, size, conc
dataset_lookup[compass]="\"'16','11','7','14'\""        # Width, size, conc
dataset_lookup[pol]="\"'1','2'\""        # Width, size, conc


DATASETS=(
adult
shoppers
default
beijing
compass
pol
)

ITER_METHODS=(
  # NPO baseline
  feature_npo_baseline
  # V/NPO
  feature_npo
  # Score orthogonal
  feature_orth
  # Gradient Difference
  feature_diff
  # Gradient Ascent
  feature_ascent
  # Feature shuffle
  feature_shuffle_v2
)

for method in "${ITER_METHODS[@]}" ; do
	for DATASET in "${DATASETS[@]}" ; do
    		SWEEP_COLUMNS=${dataset_lookup[$DATASET]}

    		echo "Submitting ${method} with ${DATASET}, with ${SWEEP_COLUMNS}"

    		sbatch --export ALL,DATASET=$DATASET,UNLEARN_METHOD=$method,SWEEP_COLUMNS=${SWEEP_COLUMNS} slurm/sweep_tabdiff.sh
	done
done
