---
layout: post
title:  "Python DataScientist Tools"
date:   2016-10-17 01:00:00
categories: "python"
asset_path: /assets/posts2/Language/
tags: ['format', 'Jupyter', 'Scipy', 'Numpy']

---


# 1. Python Libraries 

## 1.1 Dependencies

```bash
$ sudo apt install nodejs npm cmake
```

## 1.1 Basic Data Scientist Tools

Basic Libraries 

```bash
$ pip install --upgrade pip numpy pandas sklearn matplotlib
```

LightGBM

```bash
$ git clone --recursive https://github.com/microsoft/LightGBM
$ cd LightGBM
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```

## 1.2 Jupyter 

```bash
# Jupyter Lab
$ pip install jupyterlab black isort jupyterlab-code-formatter
$ jupyter server extension enable --py jupyterlab_code_formatter

# PyENV -> 버젼 찾고 커널 생성
$ pyenv versions
$ python -m ipykernel install --user --name 3.10.8 --display-name "PyEnv 3.10.8"
```