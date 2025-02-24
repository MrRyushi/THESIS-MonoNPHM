#!/bin/bash
#SBATCH -J EN2-L-512
#SBATCH --partition=gpu
#SBATCH --qos=gpu-p40_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32GB
#SBATCH -G 1
#SBATCH --job-name="mononphm"
#SBATCH --output=output_lrs3.out
#SBATCH --ntasks=1

echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

module load anaconda

source ~/scratch1/miniconda3/etc/profile.d/conda.sh
conda activate mononphm

cd scripts/preprocessing

DATASET_DIR="../../tracking_input"

# Loop through each folder inside DATASET_DIR that starts with "lrs3"
for folder in "$DATASET_DIR"/lrs3*/; do
    seq_name=$(basename "$folder")
    echo "Processing: $seq_name"
    bash run.sh "$seq_name" --no-intrinsics_provided
done

