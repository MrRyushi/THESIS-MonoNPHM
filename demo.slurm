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

python scripts/inference/rec.py --model_type nphm --exp_name pretrained_mononphm --ckpt 2500 --seq_name FFHQ_ID --intrinsics_provided --downsample_factor 0.33 --no-is_video
