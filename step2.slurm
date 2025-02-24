#!/bin/bash
#SBATCH -J EN2-L-512
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu-a100_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32GB
#SBATCH -G 1
#SBATCH --job-name="mononphm"
#SBATCH --output=output.out
#SBATCH --ntasks=1

echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

module load anaconda

source ~/scratch1/miniconda3/etc/profile.d/conda.sh
conda activate mononphm

cd scripts/preprocessing

DATASET_DIR="../../tracking_input"

# Loop through each folder inside DATASET_DIR
for folder in "$DATASET_DIR"/*/; do
    seq_name=$(basename "$folder")

    # Extract only the numeric prefix before the first underscore
    prefix=$(echo "$seq_name" | cut -d'_' -f1)

    # Check if the prefix starts with "63"
    if [[ "$prefix" == 63* ]]; then
        # Compare as a string to avoid number conversion issues
        if [[ "$prefix" > "634138" ]]; then
            echo "Processing: $seq_name"
            bash run.sh "$seq_name" --no-intrinsics_provided
        else
            echo "Skipping: $seq_name (prefix $prefix is below 633500)"
        fi
    else
        echo "Ignoring: $seq_name (prefix $prefix does not start with 63)"
    fi
done
