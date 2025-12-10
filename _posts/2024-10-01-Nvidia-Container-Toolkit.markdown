---
layout: post
title:  "Nvidia Container Toolkit"
date:   2024-10-01 01:00:00
categories: "nvidia"
asset_path: /assets/images/
tags: ['docker', 'nvidia', 'ros2', 'robotistics']
---


# Installing Nvidia Container Toolkit

아래의 링크를 참고하여 설치합니다.

- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

추가적인 설치를 다음과 같이 진행합니다. 


```bash
$ sudo apt install nvidia-container-toolkit
$ sudo apt install nvidia-container-runtime

# Docker Runtime 설정
$ sudo nvidia-ctk runtime configure --runtime=docker

# 재시작
$ sudo systemctl restart docker
```

다음과 같이 확인합니다. 

```bash
# nvidia 글자가 보이면 성공
$ docker info | grep -i runtime

# 실제 container에서 GPU를 사용할 수 있는지 확인
$ docker run --rm --gpus all ubuntu nvidia-smi
$ docker run --rm --gpus all --runtime=nvidia ubuntu nvidia-smi


```