#!/bin/bash

## create conda environment
conda create -n omnibench python=3.11 -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install "ffmpeg" -c conda-forge
conda install -c conda-forge cuda-nvrtc cuda-cudart cuda-npp

pip install nvidia-npp-cu12

pip install pandas tqdm openai pillow datasets torchcodec==0.1 
pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
pip install google-generativeai

export LD_LIBRARY_PATH=/home/t-qimhuang/miniconda3/envs/omnibench/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

