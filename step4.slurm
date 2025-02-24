#!/bin/bash
#SBATCH -J EN2-L-512
#SBATCH --partition=gpu
#SBATCH --qos=gpu-p40_default
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

# Define the directory containing the folders
DATASET_DIR="/tracking_input"

# Loop through each folder inside DATASET_DIR
for folder in "$DATASET_DIR"/*/; do
    # Extract just the folder name
    seq_name=$(basename "$folder")
    
    echo "Running inference for: $seq_name"
    
    # Run the Python script with the folder name as seq_name
    srun python scripts/inference/rec.py --model_type nphm --exp_name pretrained_mononphm_original --ckpt 2500 --seq_name "$seq_name" --is_video --no-intrinsics-provided --is_stage2
done
