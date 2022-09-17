---
layout: post
title:  "Kernelspec for PyEnv in JupyterLab"
date:   2022-08-07 01:00:00
categories: "python"
asset_path: /assets/images/
tags: []
---

# 1. Create Kernelspec for PyEnv 


## 1.1 Choose specific pyenv version 

먼저 특정 버젼의 pyenv 또는 virtualenv 를 선택합니다. 

```bash
$ pyenv global 3.10.7
$ pyenv versions
  system
  2.7.18
* 3.10.7 (set by PYENV_VERSION environment variable)
```


## 1.2 Create Kernelspec

```bash
python -m ipykernel install --user --name 3.10.7 --display-name "PyEnv 3.10.7"
```

- `--user`: 현재 유저 계정의 위치해 있는 python 위치에 설치
- `--name`: virtualenv 이름 또는 pyenv version 의 이름을 지정
- `--display-name`: Jupyter 에서 보여질 이름 지정


## 1.3 Run Jupyter Lab

```bash
$ jupyter lab
```

커널 선택시 만들어진 커널을 선택 할 수 있습니다.

<img src="{{ page.asset_path }}select-kernel-pyenv-jupyterlab.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">