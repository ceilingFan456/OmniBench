import requests
import torch
import os
import io
from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen

## reference from original repo. 
## https://huggingface.co/microsoft/Phi-4-multimodal-instruct
## https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_inference_phi4mm.py

## prompt format
# <|user|><|image_1|><|audio_1|><|end|><|assistant|>

# prompt = f'{user_prompt}<|image_1|><|audio_1|>{prompt_suffix}{assistant_prompt}'
# url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
# print(f'>>> Prompt\n{prompt}')
# image = Image.open(requests.get(url, stream=True).raw)
# audio = soundfile.read(AUDIO_FILE_1)
# inputs = processor(text=prompt, images=[image], audios=[audio], return_tensors='pt').to('cuda:0')
# generate_ids = model.generate(
#     **inputs,
#     max_new_tokens=1000,
#     generation_config=generation_config,
# )
# generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
# response = processor.batch_decode(
#     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )[0]
# print(f'>>> Response\n{response}')


## notes on the image and audio formats:
# Vision
# Any common RGB/gray image format (e.g., (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")) can be supported.
# Resolution depends on the GPU memory size. Higher resolution and more images will produce more tokens, thus using more GPU memory. During training, 64 crops can be supported. If it is a square image, the resolution would be around (8448 by 8448). For multiple-images, at most 64 frames can be supported, but with more frames as input, the resolution of each frame needs to be reduced to fit in the memory.

# Audio
# Any audio format that can be loaded by soundfile package should be supported.
# To keep the satisfactory performance, maximum audio length is suggested to be 40s. For summarization tasks, the maximum audio length is suggested to 30 mins.


# Define model path
model_path = "microsoft/Phi-4-multimodal-instruct"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True,
    # if you do not use Ampere or later GPUs, change attention to "eager"
    _attn_implementation='flash_attention_2',
).cuda()

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_path)

# Define prompt structure
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

# Part 1: Image Processing
print("\n--- IMAGE PROCESSING ---")
image_url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
prompt = f'{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')

# Download and open image
image = Image.open(requests.get(image_url, stream=True).raw)
## image format is PIL image. 
inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

# Generate response
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

# Part 2: Audio Processing
print("\n--- AUDIO PROCESSING ---")
# audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
speech_prompt = "Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the original transcript and the translation."
prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')

# Downlowd and open audio file
# audio, samplerate = sf.read(io.BytesIO(urlopen(audio_url).read()))
audio_path = "/home/t-qimhuang/code/OmniBench/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
audio, samplerate = sf.read(audio_path)

# Process with the model
## audio format is numpy array, and samplerate is int.
inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')
