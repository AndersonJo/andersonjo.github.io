---
layout: post
title:  "Nvidia RTX 6000 Pro Blackwell Workstation Settings"
date:   2026-01-29 01:00:00
categories: "format"
asset_path: /assets/images/
tags: ['pytorch', 'tensorflow', 'cuda']
---

# 1. Install Easy CUDA

We're going to install Pytorch on CUDA-13.0<br> 
depending on version, you need to run either cu12 or cu13. 


```bash
# for CUDNN 12 & 13
pip install nvidia-cudnn-cu12 \
            nvidia-cublas-cu12 \
            nvidia-cufft-cu12 \
            nvidia-curand-cu12 \
            nvidia-cusolver-cu12 \
            nvidia-cusparse-cu12 \
            nvidia-nccl-cu12

# for CUDNN 13
pip install nvidia-cudnn-cu13 \
            nvidia-cublas \
            nvidia-cufft \
            nvidia-curand \
            nvidia-cusolver \
            nvidia-cusparse \
            nvidia-nccl-cu13
```


modify ~/.bashrc

NVIDIA_HOME is different, depending on your python version


```bash
# CUDA
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




# 2. Install Pytorch

go to https://pytorch.org and find your matched version. 
 - PyTorch Build: Latest Stable
 - CUDA 13.0


create `requirements.txt`

```txt
--index-url https://pypi.org/simple
--extra-index-url https://download.pytorch.org/whl/cu130
nvidia-cudnn-cu13
nvidia-cublas
nvidia-cufft
nvidia-curand
nvidia-cusolver
nvidia-cusparse
nvidia-nccl-cu13
torch 
torchvision 

```

```bash
$ uv pip install -r requirements.txt --system
```


If you want to install `torch` independently, run it like this.

```bash
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

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




# 2. Install Unsloth

- my conclusion, Unsloth doesn't work with Tensorflow Nightly version. 
- I decided not to install tensorflow in system level (I will use Tensorflow in virtualenv)


```bash
$ pip install unsloth
#$ pip install tf-keras transformers --no-deps

# must install it again to upgrade torch and torchvision. 
# as of this writing, torch==2.10.0, torchao==0.15.0, torchaudio==2.9.1, torchvision==0.25.0
# these versions work well with unsloth==2026.1.4, unsloth_zoo==2026.1.4
$ pip install --upgrade torch torchvision
```

테스트 

```python
import torch
from unsloth import FastLanguageModel

def test_unsloth():
    # 설정
    max_seq_length = 2048
    dtype = None # None으로 설정 시 자동으로 bfloat16 (RTX 6000 지원) 감지
    load_in_4bit = True # Unsloth의 핵심인 4bit QLoRA 로딩 테스트
    
    print(f"🔹 GPU Check: {torch.cuda.get_device_name(0)}")
    print(f"🔹 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 1. 모델 및 토크나이저 로드 테스트
    print("\n[1/3] Loading Llama-3.2-1B model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Llama-3.2-1B-Instruct", 
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    # 2. Inference 테스트 (FastLanguageModel 최적화 동작 확인)
    print("\n[2/3] Running Inference...")
    FastLanguageModel.for_inference(model) # Native 2x faster inference
    
    inputs = tokenizer(
        [
            "unsloth 라이브러리의 주요 장점은 무엇인가요? 짧게 요약해주세요."
        ], return_tensors = "pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    decoded_output = tokenizer.batch_decode(outputs)[0]
    
    # 결과 출력 (내용보다는 에러 없이 생성되었는지가 중요)
    print("✅ Inference completed.")
    
    # 3. LoRA 어댑터 부착 테스트 (학습 준비 상태 확인)
    print("\n[3/3] Testing LoRA Adapter attachment...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = True,
        )
        print(f"✅ LoRA Adapters attached. Trainable parameters: {model.print_trainable_parameters()}")
    except Exception as e:
        print(f"❌ LoRA attachment failed: {e}")

test_unsloth()
```



# 3. Install Tensorflow

 - you need to install latest version of tensorflow "tf-nightly" with cuda option.
 - Also it doesn't work with `Unsloth` well.
 - I decided not to install Tensorflow nightly version on system
 - instead, I will use tensorflow in virtualenv

```bash
# Create tensorflow env
$ pyenv virtualenv tensroflow-nightly
$ pyenv activate tensroflow-nightly

# Install tensorflow nightly version with cuda support
$ pip install "tf-nightly[and-cuda]"
#
# 테스트
$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

modify ~/.bashrc

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



# 4. Install vLLM

**WARNING! here we can't install both torch and vllm at the same time!**

when you need to run, you need to run in virtualenv. 
if you install both torch and vllm, vllm downgrade your torch -> the downgraded torch will not work on RTX 6000 PRO.

```bash
# Run in virtualenv
$ pyenv virtualenv 3.12.10 vllm
$ pyenv activate vllm
$ pip install vllm 

```

```bash
python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-20b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --trust-remote-code \
    --max-model-len 5096
```




# Stable Diffusion WebUI


```bash
$ export STABLE_DIFFUSION_REPO=https://github.com/joypaul162/Stability-AI-stablediffusion.git
$ ./webui.sh
```
