---
layout: post
title:  "Nvidia RTX 6000 Pro Blackwell Workstation 600w Settings"
date:   2026-01-29 01:00:00
categories: "format"
asset_path: /assets/images/
tags: ['pytorch', 'tensorflow', 'cuda']
---

# 1. Install Pytorch

It's better to install Pytorch preview (nightly) version.<br>
I use CUDA Version 13.1

go to https://pytorch.org and find your matched version. 
 - PyTorch Build: Preview (Nightly)
 - CUDA 13.0


```bash
$ pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# check pytorch version
$ python -c "import torch; print(torch.__version__)"
```

here you can test it like this

```python
import time
import os

# Suppress TF logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def test_pytorch():
    print("--- PyTorch Check ---")
    import torch
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Detected: {gpu_name} ({vram:.2f} GB VRAM)")
        
        # Simple Compute Test
        x = torch.randn(5000, 5000, device=device)
        y = torch.randn(5000, 5000, device=device)
        start = time.time()
        result = torch.matmul(x, y)
        torch.cuda.synchronize() # Wait for compute to finish
        print(f"✅ Matrix Mul (5k x 5k) Time: {time.time() - start:.4f}s")
    else:
        print("❌ PyTorch cannot see the GPU.")

test_pytorch()
```


# 2. Tensorflow

 - you need to install latest version of tensorflow "tf-nightly" with cuda option.

firstly, modify ~/.bashrc

```bash
# CUDA for Tensorflow
NVIDIA_HOME="$HOME/.pyenv/versions/3.12.10/lib/python3.12/site-packages/nvidia"

# 2. 필요한 라이브러리 경로들을 LD_LIBRARY_PATH에 추가
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cublas/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cufft/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/curand/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cusolver/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cusparse/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/nccl/lib
```

```bash
$ pip install "tf-nightly[and-cuda]"

# 테스트
$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```


if we use \[and-cuda\], it automatically downloads cudnn, cublas, ... nccl.<br>
However if you want to install all of them manually, you can do it like this. 

```bash
pip install nvidia-cudnn-cu12 \
            nvidia-cublas-cu12 \
            nvidia-cufft-cu12 \
            nvidia-curand-cu12 \
            nvidia-cusolver-cu12 \
            nvidia-cusparse-cu12 \
            nvidia-nccl-cu12
```


here's a bit more complex test. 

```python
import time
import os

def test_tensorflow():
    print("\n--- TensorFlow Check ---")
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   - {gpu.device_type}: {gpu.name}")
            
        # Simple Compute Test
        with tf.device('/GPU:0'):
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            start = time.time()
            c = tf.matmul(a, b)
            _ = c.numpy() # Force execution
            print(f"✅ Matrix Mul (5k x 5k) Time: {time.time() - start:.4f}s")
    else:
        print("❌ TensorFlow cannot see the GPU.")


test_tensorflow()
```

# 3. vLLM

 - here, we need to install nightly version

```python
$ pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

```bash
python -m vllm.entrypoints.openai.api_server \
    --model moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --served-model-name kimi-k2.5 \
    --max-model-len 32768
    
```



# Stable Diffusion WebUI

 - in venv, you need to install pytorch nightly version.

```bash
$ export STABLE_DIFFUSION_REPO=https://github.com/joypaul162/Stability-AI-stablediffusion.git
$ ./webui.sh
```
