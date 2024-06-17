---
layout: post
title: "Macbook - Tensorflow on M1 Macbook"
date:  2022-09-15 01:00:00
categories: "format"
asset_path: /assets/images/
tags: ['brew', 'arm64', 'x86_64', 'intel', 'docker', 'm1', 'macbook']
---


# Installing Tensorflow on M1 Macbook

## Prerequisites

```bash
brew install openblas gfortran lapack
```

~/.bash_profile 에 다음을 추가 합니다.

```bash
export# openblas
export LDFLAGS="-L/usr/local/opt/openblas/lib"
export CPPFLAGS="-I/usr/local/opt/openblas/include"
export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig"

# lapack
export LDFLAGS="-L/usr/local/opt/lapack/lib"
export CPPFLAGS="-I/usr/local/opt/lapack/include"
export PKG_CONFIG_PATH="/usr/local/opt/lapack/lib/pkgconfig"
```

Python Libraries 를 설치합니다. 


## Install Conda Environment

[apple_tensorflow_20220828](https://github.com/PeterSels/OnComputation/blob/master/TensorFlowM1/apple_tensorflow_20220828.yml)을 다운로드 합니다.

```
$ conda config --set auto_activate_base false
$ conda env create --file=~/Downloads/apple_tensorflow_20220828.yml --name=apple_tf
$ conda activate apple_tf
```


## Install Miniforge 

[Miniforge Github](https://github.com/conda-forge/miniforge#miniforge3) 에 들어가서 OS X (Apple Silicon)을 다운로드 받습니다. 

- [Miniforge3-MacOSX-arm64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)

```bash
chmod a+x ~/Downloads/Miniforge3-MacOSX-arm64.sh
~/Downloads/Miniforge3-MacOSX-arm64.sh
```


> PyENV 사용시 .bash_profile 에서 >>> conda initialization 부분을 전부다 comment out 합니다.

아래 코드를 .bash_profile 에 추가합니다. 


```bash
alias alaconda=~/miniforge3/condabin
alias miniforge=~/miniforge3/condabin/conda
```

## Install Tensorflow 


```bash
# Env 체크
$ conda activate apple_tf

# 관련 파일 설치
$ pip install --upgrade pip
$ pip install tensorflow-macos tensorflow-metal
$ pip install protobuf==3.19.6
```




