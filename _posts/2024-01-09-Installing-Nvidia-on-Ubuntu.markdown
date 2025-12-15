---
layout: post
title:  "Nvidia Driver on Ubuntu"
date:   2024-01-09 01:00:00
categories: "nvidia"
asset_path: /assets/images/
tags: ['format']
---

## Prerequisite

- Ubuntu 설치시에 update 하지 않는 것도 방법 (깔고나서 에러 나면. linux-headers 가 문제)
- Secure Boot 모드는 disabled 시켜놓자 (설치할때 password 넣으라고 하는데 귀찮음)

```bash
$ sudo apt install make gcc vim openssl libgoogle-perftools4 libtcmalloc-minimal4 g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
$ sudo apt install linux-headers-generic
$ sudo apt install libglu1-mesa libxi-dev libxmu-dev gcc build-essential
```

## Uninstalling Nvidia Driver

```bash
$ sudo apt-get remove --purge 'nvidia.*'
$ sudo apt-get remove --purge 'cuda.*'
$ sudo apt-get remove --purge 'libnvidia*'
$ sudo apt-get autoremove
$ sudo apt autoclean
$ sudo apt-get install ubuntu-desktop
$ sudo rm /etc/X11/xorg.conf
$ sudo nvidia-uninstall
```

## Checking Current Nvidia Driver

현재 설치된 Nvidia 버젼을 확인합니다.

```bash
$ modinfo $(find /usr/lib/modules -name nvidia.ko)

# apt 로 확인
$ sudo apt --installed list | grep nvidia-driver
```

## Checking Supported CUDA Version





## Installing Nvidia Driver 

일단 **설치 가능한 버젼**을 확인합니다. 

```bash
$ sudo ubuntu-drivers list --gpgpu

nvidia-driver-570-server-open, (kernel modules provided by linux-modules-nvidia-570-server-open-generic-hwe-24.04)
nvidia-driver-580-server-open, (kernel modules provided by linux-modules-nvidia-580-server-open-generic-hwe-24.04)
nvidia-driver-535-server, (kernel modules provided by linux-modules-nvidia-535-server-generic-hwe-24.04)
nvidia-driver-570-server, (kernel modules provided by linux-modules-nvidia-570-server-generic-hwe-24.04)
nvidia-driver-535, (kernel modules provided by linux-modules-nvidia-535-generic-hwe-24.04)
nvidia-driver-580-server, (kernel modules provided by linux-modules-nvidia-580-server-generic-hwe-24.04)
nvidia-driver-570-open, (kernel modules provided by linux-modules-nvidia-570-open-generic-hwe-24.04)
nvidia-driver-580-open, (kernel modules provided by linux-modules-nvidia-580-open-generic-hwe-24.04)
nvidia-driver-535-server-open, (kernel modules provided by linux-modules-nvidia-535-server-open-generic-hwe-24.04)
nvidia-driver-570, (kernel modules provided by linux-modules-nvidia-570-generic-hwe-24.04)
nvidia-driver-535-open, (kernel modules provided by linux-modules-nvidia-535-open-generic-hwe-24.04)
nvidia-driver-580, (kernel modules provided by linux-modules-nvidia-580-generic-hwe-24.04)
```

3090 에서 내 컴퓨터에서 돌아가는 버젼은 nvidia-driver-570-open

이후 설치 합니다. <br>
2024년 6월 기준으로 **520** 에다가 CUDA Toolkit 11.8 이 잘 작동합니다. (Ubuntu 22.04)<br>
545 아래는 에러가 났습니다. 

```bash
# 커널 설치
$ sudo apt install linux-headers-$(uname -r)

# Nvidia 드라이버 설치
# xxx 부분은 예를 들어서 "570-open" 등으로 넣어줍니다.
$ sudo apt install nvidia-driver-xxx
$ sudo apt install nvidia-settings
```

이후 설치된 package 를 확인 할 수 있습니다.

```bash
$ dpkg --get-selections | grep nvidia
$ dpkg --get-selections | grep cuda
```

## Secure Boot

Secure Boot 이 켜져 있으면 드라어버가 정상적으로 로드가 안될수 있습니다. <br>
이 경우에는 다음을 실행합니다. 

```bash
sudo update-secureboot-policy --enroll-key
```

암호 쓰라고 나오고, 암호 쓰고 재부팅 하면, 이전에 썼던 암호를 다시 쓰고, MOK enrollment를 설정합니다. <br>


## Installing CUDA Toolkit

```bash
# CUDA 툴킷 설치
sudo apt install nvidia-cuda-toolkit

# 버젼 확인
nvcc --version
```


## Install CuDNN

 - [Download CuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)


```bash
$ wget https://developer.download.nvidia.com/compute/cudnn/9.1.1/local_installers/cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb
$ sudo dpkg -i cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb
$ sudo cp /var/cudnn-local-repo-ubuntu2204-9.1.1/cudnn-*-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update

# 설치합니다. 
$ sudo apt-get -y install cudnn-cuda-11

# 버젼 확인합니다. 
$ nvcc --version
Build cuda_11.5.r11.5/compiler.30672275_0
```

## Install Pytorch TensorRT

 - [Installing TensorRT](https://pytorch.org/TensorRT/getting_started/installation.html)

cuda 11.5 에는 다음을 설치 합니다. 
```bash
$ python -m pip install torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu115
```

cuda 11.8 에는 다음을 설치 합니다.
```bash
$ python -m pip install torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu118
```

잘 작동하는지 확인 합니다. 

```python
import torch
import torch_tensorrt

# 간단한 모델 정의 (스크립팅 가능)
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x + 1

model = SimpleModel().cuda().eval()

# JIT 스크립트 모듈로 변환
scripted_model = torch.jit.script(model)

# TensorRT 변환
input_tensor = torch.randn((1, 3, 224, 224)).cuda()
trt_model = torch_tensorrt.ts.compile(
    scripted_model,
    inputs=[torch_tensorrt.Input(input_tensor.shape)]
)

# 변환된 모델로 추론
with torch.no_grad():
    output = trt_model(input_tensor)

print("TensorRT 변환 및 추론 성공:", output.shape)
```


## Disable Nouveau

기존 우분투에서 지원하는 그래픽 드라이버를 제거합니다.
Nvidia 그래픽 드라이버와 서로 충돌이 나면서 이후 문제가 생기는 것을 방지 합니다.

```bash
$ sudo vi  /etc/modprobe.d/nouveau-blacklist.conf 
```

/etc/modprobe.d/nouveau-blacklist.conf 에 아래의 내용을 넣습니다.

```bash
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```

이후 다음의 명령어로 부팅을 업데이트 해줍니다.


``` bash
sudo update-initramfs -u
```

이후 reboot 시킵니다.


## Test

**Tensorflow**
```bash
$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Pytorch**

```bash
import torch
x = torch.rand(5, 3)
print(x)
torch.cuda.is_available()
```