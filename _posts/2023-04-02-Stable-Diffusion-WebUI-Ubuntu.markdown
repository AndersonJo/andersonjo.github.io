---
layout: post
title:  "Stable Diffusion WebUI on Ubuntu"
date:   2023-04-02 01:00:00
categories: "generative-ai"
asset_path: /assets/images/
tags: []
---

# WebUI

## 1. Preparation

 - Python: 3.10.6
 - Install location: ~/projects/

```bash
$ pyenv install 3.10.6
$ git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
$ cd ~/projects/stable-diffusion-webui/
$ pyenv local 3.10.6
```

## Running 

자동으로 venv 만들고, 라이브러리 알아서 설치하기 때문에 따로, 만들 필요가 없습니다. 

```bash
$ ./webui.sh
```

http://localhost:7860/


# 2. 유용한 모델

| Category         | Model                                                                                                                                               | Description |
|:-----------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:------------|
| Stable Diffusion | [chilloutmix_NiPrunedFp32Fix.safetensors(https://civitai.com/models/6424/chilloutmix)                                                               |             |
|                  | [yaeMikoRealistic_yaemikoMixed](https://huggingface.co/SakerLy/yaeMikoRealistic_yaemikoMixed/blob/main/yaeMikoRealistic_yaemikoMixed.safetensors)   |             | 
