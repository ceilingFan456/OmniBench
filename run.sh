#!/bin/bash

export LD_LIBRARY_PATH=/home/t-qimhuang/miniconda3/envs/omnibench_py310/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/t-qimhuang/miniconda3/envs/omnibench_py310/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH



cd /home/t-qimhuang/code/OmniBench
# python demo_api_call.py --output-file outputs/test_inference_output.json
python demo_api_call_seq.py --model-name-or-path phi4_local --output-file outputs/test_inference_output.json
