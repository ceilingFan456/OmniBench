#!/bin/bash
set -e

# ENV=omnibench_py310

# # 1) Create env
# conda create -n ${ENV} python=3.10 -y
# conda activate ${ENV}

# 2) PyTorch (match your CUDA wheel line; cu121 is fine with your driver)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 3) System/media libs (ffmpeg for audio/video decoding tools; harmless even if you skip decoding)
# conda install -c conda-forge -y ffmpeg

# 4) Phi-4 suggested stack (your list)
pip install psutil

pip install --no-build-isolation flash_attn==2.7.4.post1 

pip install transformers==4.48.2 \
  accelerate==1.3.0 \
  soundfile==0.13.1 \
  pillow==11.1.0 \
  scipy==1.15.2 \
  backoff==2.2.1 \
  peft==0.13.2

# 5) OmniBench runtime deps
pip install pandas tqdm datasets openai

# 6) Optional (only if you truly need these vendor clients)
pip install google-generativeai anthropic || true

# 7) CUDA runtime shared libs needed by some multimedia stacks (torchcodec / etc.)
# NVRTC + runtime
pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
# NPP (for libnppicc.so.12)
pip install nvidia-npp-cu12

# pip install vllm

# 8) Export library paths so dlopen() can find .so files at runtime
NVRTC_LIB=$(python - <<'PY'
import site, glob, os
paths=[]
for d in site.getsitepackages():
    paths += glob.glob(os.path.join(d,"**","libnvrtc.so*"), recursive=True)
print(os.path.dirname(paths[0]) if paths else "")
PY
)

NPP_LIB=$(python - <<'PY'
import site, glob, os
paths=[]
for d in site.getsitepackages():
    paths += glob.glob(os.path.join(d,"**","libnppicc.so*"), recursive=True)
print(os.path.dirname(paths[0]) if paths else "")
PY
)

if [ -n "$NVRTC_LIB" ]; then
  export LD_LIBRARY_PATH="$NVRTC_LIB:$LD_LIBRARY_PATH"
fi
if [ -n "$NPP_LIB" ]; then
  export LD_LIBRARY_PATH="$NPP_LIB:$LD_LIBRARY_PATH"
fi

echo "Activated env: $ENV"
python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__)"
echo "LD_LIBRARY_PATH includes:"
echo "$NVRTC_LIB"
echo "$NPP_LIB"
