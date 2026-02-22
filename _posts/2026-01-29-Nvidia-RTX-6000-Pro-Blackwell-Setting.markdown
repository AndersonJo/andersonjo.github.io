---
layout: post
title:  "Nvidia RTX 6000 Pro Blackwell Workstation Settings"
date:   2026-01-29 01:00:00
categories: "format"
asset_path: /assets/images/
tags: ['pytorch', 'tensorflow', 'cuda', "continue"]
---





# 1. Install Pytorch


## 1.1 Pytorch with CUDA 13.0

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
transformers
```

```bash
$ uv pip install -r requirements.txt --system
```


## 1.2 Pytorch with CUDA 12.8


create `requirements.txt`


```txt
--index-url https://pypi.org/simple
--extra-index-url https://download.pytorch.org/whl/cu128
nvidia-cudnn-cu12
nvidia-cublas-cu12
nvidia-cufft-cu12
nvidia-curand-cu12
nvidia-cusolver-cu12
nvidia-cusparse-cu12
nvidia-nccl-cu12
torch 
torchvision 
transformers
```

또는 

```txt
uv pip install nvidia-cudnn-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12 \
    torch \
    torchvision \
    transformers \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

```bash
$ uv pip install -r requirements.txt --system
```


## 1.3 setting .bashrc 


modify ~/.bashrc<br>
NVIDIA_HOME is different, depending on your python version


```bash
# CUDA
NVIDIA_HOME="$HOME/.pyenv/versions/3.12.10/lib/python3.12/site-packages/nvidia"

# 2. Add necessary libraries into LD_LIBRARY_PATH에
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cu13/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cublas/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cufft/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/curand/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cusolver/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/cusparse/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVIDIA_HOME}/nccl/lib
```


## 1.4 Test

```bash
$ python -c "import torch; print(torch.__version__)"
```


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


# 3. Install TLR

```python
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, SFTConfig, DPOConfig
import os
import tempfile
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore")


def print_status(message):
    print(f"\n[TRL TEST] >> {message}")


def run_trl_health_check():
    print_status("Starting TRL Health Check...")

    # 1. Environment & Device Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_status(f"Device: {device}")

    # 2. Setup Resources (Correct Vocab Size Match)
    print_status("Initializing dummy model and tokenizer...")

    # 먼저 토크나이저를 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 모델 설정: vocab_size를 토크나이저와 동일하게 맞춤 (매우 중요)
    model_config = AutoConfig.from_pretrained("gpt2")
    model_config.n_layer = 2
    model_config.n_head = 2
    model_config.n_embd = 32
    model_config.vocab_size = len(tokenizer)  # <--- 여기가 수정되었습니다. (50257)

    model = AutoModelForCausalLM.from_config(model_config).to(device)

    # 3. Test SFTTrainer (Supervised Fine-Tuning)
    print_status("Testing SFTTrainer (1 training step)...")

    sft_data = {
        "text": [
                    "User: Hello\nAssistant: Hi there!",
                    "User: Code for me\nAssistant: Sure, here is python code.",
                    "User: Bye\nAssistant: Goodbye!"
                ] * 10
    }
    sft_dataset = Dataset.from_dict(sft_data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        sft_config = SFTConfig(
            output_dir=tmp_dir,
            dataset_text_field="text",
            max_length=16,
            per_device_train_batch_size=2,
            max_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            report_to="none",
            save_strategy="no",
        )

        sft_trainer = SFTTrainer(
            model=model,
            train_dataset=sft_dataset,
            args=sft_config,
            processing_class=tokenizer,  # it was "tokenizer" previously
        )

        sft_trainer.train()
        print_status("SFTTrainer executed successfully.")

    # 4. Test DPOTrainer (Direct Preference Optimization)
    print_status("Testing DPOTrainer initialization...")

    dpo_data = {
        "prompt": ["Question 1", "Question 2"] * 5,
        "chosen": ["Good answer 1", "Good answer 2"] * 5,
        "rejected": ["Bad answer 1", "Bad answer 2"] * 5,
    }
    dpo_dataset = Dataset.from_dict(dpo_data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dpo_config = DPOConfig(
            output_dir=tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            report_to="none",
            learning_rate=1e-5,
        )

        # DPO용 새 모델 (동일 설정)
        dpo_model = AutoModelForCausalLM.from_config(model_config).to(device)

        dpo_trainer = DPOTrainer(
            model=dpo_model,
            ref_model=None,
            args=dpo_config,
            train_dataset=dpo_dataset,
            processing_class=tokenizer,  # it was "tokenizer" previously
        )

        dataloader = dpo_trainer.get_train_dataloader()
        batch = next(iter(dataloader))
        print_status("DPOTrainer initialized and data processed successfully.")

    # 5. Test PPO Integration (Value Head)
    print_status("Testing PPO ValueHead Model Wrapper...")

    try:
        # PPO 모델은 기본 모델 위에 Value Head를 얹는 구조
        ppo_base_model = AutoModelForCausalLM.from_config(model_config).to(device)
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_base_model)

        if hasattr(ppo_model, "v_head"):
            print_status("AutoModelForCausalLMWithValueHead successfully attached 'v_head'.")
        else:
            raise ValueError("v_head not found in PPO model.")

        inputs = tokenizer("Test input", return_tensors="pt").to(device)
        with torch.no_grad():
            output = ppo_model(**inputs)
        print_status("PPO Model forward pass successful.")

    except Exception as e:
        print(f"PPO Test Failed: {e}")
        # PPO is tricky with dummy models sometimes, but we proceed if minor error
        pass

    # 6. Test PEFT Integration (LoRA)
    print_status("Testing PEFT (LoRA) Integration with TRL...")

    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    base_model_peft = AutoModelForCausalLM.from_config(model_config).to(device)
    peft_model = get_peft_model(base_model_peft, peft_config)

    print_status(f"PEFT Model created. Trainable params: {peft_model.print_trainable_parameters()}")

    print_status("\n---------------------------------------------------")
    print_status("SUCCESS: All TRL components checked thoroughly.")
    print_status("---------------------------------------------------")


run_trl_health_check()
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
$ pyenv virtualenv 3.12.12 vllm
$ pyenv activate vllm
$ pip install vllm 

```

```bash
vllm serve openai/gpt-oss-20b \
    --host 127.0.0.1 \
    --port 8082 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.3 \
    --trust-remote-code \
    --async-scheduling \
    --max-num-batched-tokens 8192 \
    --max-model-len 35096
```




# 5. Stable Diffusion WebUI

 - python: 3.10.17
 - 

```bash
$ export STABLE_DIFFUSION_REPO=https://github.com/joypaul162/Stability-AI-stablediffusion.git

# Install specific setuptools
$ source ./venv/bin/activate
$ pip install https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip --no-build-isolation
$ pip install setuptools==69.5.1 wheel
$ deactivate 

# 설치
$ ./webui.sh
```


disk error issue (this is just a personal issue. just skip it)

```bash
$ sudo umount -l /dev/nvme1n1p2
$ sudo ntfsfix -d /dev/nvme1n1p2
$ sudo mount /dev/nvme1n1p2 /media/anderson/HynixP41
```



# 6. CONTINUE on Pycharm 

CONTINUE is a plugin for llm in Pycharm. 


config.yaml


```aiexclude
name: Local Config
version: 1.0.0
schema: v1
models:
  - name: Llama 3.1 8B
    provider: ollama
    model: llama3.1:8b
    roles:
      - chat
      - edit
      - apply
  - name: Qwen2.5-Coder 1.5B
    provider: ollama
    model: qwen2.5-coder:1.5b-base
    roles:
      - autocomplete
  - name: Nomic Embed
    provider: ollama
    model: nomic-embed-text:latest
    roles:
      - embed
  - name: Qwen3-Coder-30B (Local)
    provider: openai
    model: Qwen/Qwen3-Coder-30B-A3B-Instruct
    apiBase: http://localhost:8045/v1
#    apiKey: my-secret-key
    roles:
      - chat
      - edit
      - apply
      - autocomplete
  - name: Nomic Embed
    provider: ollama
    model: nomic-embed-text:latest
    roles:
      - embed
```

you can create vllm

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --dtype auto \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --max-model-len 50000 \
    --port 8045
```