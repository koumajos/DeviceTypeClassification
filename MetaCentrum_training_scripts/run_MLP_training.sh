
#!/bin/bash
#PBS -N CESNET-TimeSeries24_classification
#PBS -l select=1:ncpus=1:ngpus=0:mem=16gb:scratch_local=14gb
#PBS -l walltime=24:00:00


cleanup() {
    echo "Cleaning up SCRATCHDIR at $SCRATCHDIR"
    cp "$SCRATCHDIR"/*.pth /storage/plzen1/home/mudrukar/cesnet_time_series_neural_nets/mlp_models/
    cp "$SCRATCHDIR"/*.txt /storage/plzen1/home/mudrukar/cesnet_time_series_neural_nets/mlp_results/
    rm -rf "$SCRATCHDIR"/*
}
trap cleanup EXIT

echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

# Load venv
HOMEDIR=/storage/brno2/home/$USER
SYNCED=$HOMEDIR/synced
HOSTNAME=`hostname -f`
source $SYNCED/venv/bin/activate


# Copy data to scratch
echo "Copying data to $SCRATCHDIR"
PREPROCESSING=$PREPROCESSING

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

cp $SRC_TRAIN/group_*.npy "$SCRATCHDIR" || { echo >&2 "Error while copying training files!"; exit 2; }
cp $SRC_VAL/*.npy "$SCRATCHDIR" || { echo >&2 "Error while copying validation files!"; exit 2; }

python /storage/plzen1/home/mudrukar/cesnet_time_series_neural_nets/run_MLP_training.py\
    --hidden_size $HIDDEN_SIZE\
    --activation_f $ACTIVATION_F\
    --opt adam \
    --lr $LR \
    --num_layers $NUM_LAYERS \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --preprocessing $PREPROCESSING \
    --scratch_dir "$SCRATCHDIR"
