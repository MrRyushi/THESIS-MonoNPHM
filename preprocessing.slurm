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

#conda env remove --name mononphm -y
#conda env create -f environment.yml  
source ~/scratch1/miniconda3/etc/profile.d/conda.sh
conda activate mononphm
conda info | grep "active environment"

#echo $LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#pip uninstall torch torchvision torchaudio -y
#pip cache purge
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#pip uninstall torch torchvision -y
#pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 


#conda remove pytorch torchvision torchaudio -y
#conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Install pytorch with CUDA support
#conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Install PytorchGeometry and helper packages with CUDA support
#conda install pyg -c pyg -y
#pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html 

# Install Pytorch3D with CUDA support
#conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
#conda install pytorch3d=0.7.4 -c pytorch3d -y

#pip uninstall numpy -y
#pip install numpy==1.23
#pip install pyopengl==3.1.5

#pip install -e .
cd scripts/preprocessing
bash run.sh 6330311066473698535_00011 --no-intrinsics_provided

#conda remove pytorch torchvision torchaudio pyg -y
#conda remove pytorch-mutex -y
#conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
#pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
#conda env export | grep torch
#pip uninstall numpy -y
#pip install numpy==1.23
#pip install pyopengl==3.1.5

