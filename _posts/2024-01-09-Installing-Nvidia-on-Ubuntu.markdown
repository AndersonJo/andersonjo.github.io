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
$ sudo apt install make gcc vim openssl libgoogle-perftools4 libtcmalloc-minimal4 
$ sudo apt install linux-headers-generic
$ sudo apt install libglu1-mesa libxi-dev libxmu-dev gcc build-essential
```

## Uninstalling Nvidia Driver

```bash
$ sudo apt-get remove --purge 'nvidia.*'
$ sudo apt-get remove --purge 'cuda.*'
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

- [Pytorch supported CUDA version 확인](https://pytorch.org/get-started/locally/)
- [Tensorflow supported CUDA version 확인](https://www.tensorflow.org/install/pip?hl=ko)

CUDA Toolkit version 
 - [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

| Pytorch CUDA | Tensorflow CUDA |
|:-------------|:----------------|
| 11.8         | 11.8            |
| 12.1         |                 |

CUDNN

| Pytorch CUDA | Tensorflow CUDA | Nvidia Version |
|:-------------|:----------------|:---------------|
|              | 8.6.0           | 520            |




## Installing Nvidia Driver 

일단 **설치 가능한 버젼**을 확인합니다. 

```bash
$ sudo ubuntu-drivers list --gpgpu

nvidia-driver-470-server, (kernel modules provided by linux-modules-nvidia-470-server-generic-hwe-22.04)
nvidia-driver-535-server, (kernel modules provided by linux-modules-nvidia-535-server-generic-hwe-22.04)
nvidia-driver-535-open, (kernel modules provided by linux-modules-nvidia-535-open-generic-hwe-22.04)
nvidia-driver-470, (kernel modules provided by linux-modules-nvidia-470-generic-hwe-22.04)
nvidia-driver-535, (kernel modules provided by linux-modules-nvidia-535-generic-hwe-22.04)
nvidia-driver-545, (kernel modules provided by nvidia-dkms-545)
nvidia-driver-545-open, (kernel modules provided by nvidia-dkms-545-open)
nvidia-driver-535-server-open, (kernel modules provided by linux-modules-nvidia-535-server-open-generic-hwe-22.04)
```

이후 설치 합니다. <br>
2024년 6월 기준으로 520 에다가 CUDA Toolkit 11.8 이 잘 작동합니다. (Ubuntu 22.04)  

```bash
# 커널 설치
$ sudo apt install linux-headers-$(uname -r)

# Nvidia 드라이버 설치
# xxx 부분은 예를 들어서 "520" 
$ sudo apt install nvidia-common-xxx
$ sudo apt install nvidia-dkms-xxx
$ sudo apt install nvidia-driver-xxx
$ sudo apt install nvidia-settings
```

이후 설치된 package 를 확인 할 수 있습니다.

```bash
$ dpkg --get-selections | grep nvidia
$ dpkg --get-selections | grep cuda
```


## Installing CUDA Toolkit 11.8

- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

```bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
$ sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
$ sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get -y install cuda
```

## Installing CuDNN





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
