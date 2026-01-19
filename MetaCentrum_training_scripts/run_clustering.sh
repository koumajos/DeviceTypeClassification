#!/bin/bash
#PBS -N CESNET-TimeSeries24_histogram_clustering
#PBS -l select=1:ncpus=1:mem=100gb:scratch_local=15gb
#PBS -l walltime=12:00:00

cleanup() {
    cp "$SCRATCHDIR"/*.txt /storage/plzen1/home/mudrukar/clustering/histogram_results/
    cp "$SCRATCHDIR"/*.pkl /storage/plzen1/home/mudrukar/clustering/models/
    rm -rf "$SCRATCHDIR"/*
}
trap cleanup EXIT

echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

HOMEDIR=/storage/brno2/home/$USER
SYNCED=$HOMEDIR/synced
HOSTNAME=$(hostname -f)
source $SYNCED/venv/bin/activate

: "${N_CLUST:?Environment variable N_CLUST not set!}"
: "${PREPROCESSING:?Environment variable PREPROCESSING not set!}"



if [[ "$PREPROCESSING" == "log" ]]; then
    SRC_TRAIN="/storage/plzen1/home/mudrukar/cesnet_time_series_neural_nets/preprocessed_train_data_log"
    SRC_VAL="/storage/plzen1/home/mudrukar/cesnet_time_series_neural_nets/preprocessed_val_data_log"
elif [[ "$PREPROCESSING" == "z_score" ]]; then
    SRC_TRAIN="/storage/plzen1/home/mudrukar/cesnet_time_series_neural_nets/preprocessed_train_data_z"
    SRC_VAL="/storage/plzen1/home/mudrukar/cesnet_time_series_neural_nets/preprocessed_val_data_z"
else
    echo "Invalid preprocessing type: $PREPROCESSING"
    exit 1
fi

cp $SRC_TRAIN/group_*.npy "$SCRATCHDIR" || { echo >&2 "Error copying training files!"; exit 2; }
cp $SRC_VAL/*.npy "$SCRATCHDIR" || { echo >&2 "Error copying validation files!"; exit 2; }

DISTANCE_METRIC="euclidean"
UNDERSAMPLE_FLAG="--undersample"

python /storage/plzen1/home/mudrukar/clustering/run_histogram_clustering.py \
  --distance_metric "$DISTANCE_METRIC" \
  --n_clust "$N_CLUST" \
  --scratch_dir "$SCRATCHDIR" \
  --preprocessing "$PREPROCESSING" \
  --undersample "$UNDERSAMPLE"

