#!/bin/bash

# Need to run this first to set up the data properly
# python train_nnunet.py --train-data-folder /home/laura/Documents/VirtualFixture/uncertainty/TrainingData/train/ --val-data-folder /home/laura/Documents/VirtualFixture/uncertainty/TrainingData/val/ --output-dir /home/laura/Documents/VirtualFixture/uncertainty/output/ --save-torchscript --save-log --nnunet-dataset-name Dataset001_Breast --verify-nnunet-dataset
# copied ultrasound dataset file into current folder
# Define paths =
TRAIN_DATA_FOLDER="/home/laura/Documents/VirtualFixture/uncertainty/TrainingData/train/"
VAL_DATA_FOLDER="/home/laura/Documents/VirtualFixture/uncertainty/TrainingData/val/"
OUTPUT_DIR="/home/laura/Documents/VirtualFixture/uncertainty/ensemble_output/"
NNUNET_DATASET_NAME="Dataset001_Breast"
CONFIG_FILE="train_config.yaml"

# Ensemble size
ENSEMBLE_SIZE=5
# Array to store seeds
declare -a seeds

# Train the nnUNet 5 times with a new seed each time
for i in $(seq 1 $ENSEMBLE_SIZE)
do
    # Generate a new seed for each training session
    SEED=$(( (RANDOM % 100) + 1 )) # capped at 100
    seeds[$i]=$SEED
    # Define unique output directory for each model in the ensemble
    MODEL_OUTPUT_DIR="${OUTPUT_DIR}model_${i}/"

    echo "Training model $i with seed $SEED"

    # Update the seed in the configuration file
    sed -i "s/seed: !!int [0-9]*/seed: !!int $SEED/" $CONFIG_FILE

    # Run training script with different seeds
    python3.9 train_nnunet.py \
        --train-data-folder $TRAIN_DATA_FOLDER \
        --val-data-folder $VAL_DATA_FOLDER \
        --output-dir $MODEL_OUTPUT_DIR \
        --save-torchscript \
        --save-log \
        --nnunet-dataset-name $NNUNET_DATASET_NAME \
        --verify-nnunet-dataset \
        --config-file $CONFIG_FILE \

done

echo "Ensemble training complete."
# Print all used seeds
echo "Seeds used in the ensemble:"
for seed in "${seeds[@]}"
do
    echo $seed
done
