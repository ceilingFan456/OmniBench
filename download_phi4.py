#!/usr/bin/env python
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="microsoft/Phi-4-multimodal-instruct")
speech_lora_path = model_path+"/speech-lora"
vision_lora_path = model_path+"/vision-lora"

## /home/t-qimhuang/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81
print(f"Model path: {model_path}")
## /home/t-qimhuang/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81/speech-lora
print(f"Speech LoRA path: {speech_lora_path}")
##  /home/t-qimhuang/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81/vision-lora
print(f"Vision LoRA path: {vision_lora_path}")

## python -m vllm.entrypoints.openai.api_server --model 'microsoft/Phi-4-multimodal-instruct' --dtype auto --trust-remote-code --max-model-len 131072 --enable-lora --max-lora-rank 320 --lora-extra-vocab-size 0 --limit-mm-per-prompt audio=3,image=3 --max-loras 2 --lora-modules speech=/home/t-qimhuang/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81/speech-lora vision=/home/t-qimhuang/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81/vision-lora
