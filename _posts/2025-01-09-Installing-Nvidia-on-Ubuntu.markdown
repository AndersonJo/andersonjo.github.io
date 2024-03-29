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



## Installing Nvidia Driver 

일단 설치 가능한 버젼을 확인합니다. 
```bash
$ sudo ubuntu-drivers list --gpgpu
```

이후 설치 합니다. 

```bash
# 커널 설치
$ sudo apt install linux-headers-$(uname -r)

# Nvidia 드라이버 설치
# xxx 부분은 예를 들어서 "525"
$ sudo apt install nvidia-common
$ sudo apt install nvidia-dkms-xxx
$ sudo apt install nvidia-driver-xxx
$ sudo apt install nvidia-settings
```
