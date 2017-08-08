---
layout: page
title: 딥러닝 모델 비교 분석
date:   2017-08-05 01:00:00
categories: "deep-learning"
asset_path: /assets/images/
tags: ['resnet', 'densenet']

---

## System

비교를 하는 시스템의 상황은 다음과 같습니다.

| Name | Value |
|:-----|:------|
| CPU  | Intel I7 6700K CPU 4.00GHz, 8 cores |
| Memory | 32GB |
| GPU  | [NVIDIA GTX 1080](https://www.geforce.co.uk/hardware/desktop-gpus/geforce-gtx-1080/specifications), 8GB GDDR5X Memory |
| OS   | Ubuntu 16.04 |


## Models

| Model | Params | Lib | Data | Loss | Accuracy | Memory | CPU(T) | CPU(P) | GPU(T) | GPU(P) |
|:------|:-------|:----|:-----|:-----|:---------|:-------|:-------|:-------|:-------|:-------|
| AR(OLS)  | 12 Lag <br>1 x w, 1 x b | Numpy   | Airline Passenger | 0.05 | -0.18| | | | |
| AR(Polyfit) | 12 Lag <br>1 degree    | Numpy   | Airline Passenger | 0.01 | 0.62 | | | | |
| DenseNet | 3 x 24 dense | Pytorch | CIFAR-10 | | |  6619MB | 120 | 25.23 | 0.337 | 0.111 |


* **Models**
  - OLS : Ordinary Least Square Estimation
  - PolyFit : Polinomial Curve Fitting
  - AR : Autoregressive Model
  - DenseNet : 메모리 최적화되지 않은 [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)의 구현된 모델
* **Parameters**
  - w: weight
  - b: bias
  - convolution : Convolution Layer
  - fc : Fully Connected Layer (또는 Affine, Linear 라고 보면 됨)
* **Data**
  - CIFAR-10 : 32 x 32 이미지, 10 categories
* **BATCH** : 모든 모델의 Batch 의 크기는 32를 사용하였습니다.
* **Memory** : GPU의 메모리 사용량을 의미합니다.
* **CPU(T)** : T는 Training을 가르키며, CPU기반 학습할때 초당 1 Batch 데이터를 처리하는 속도입니다.
* **CPU(P)** : P는 Predict를 가르키며, CPU에서 학습이 완료된 모델을 갖고서 1 Batch 데이터를 추론/예측 하는 속도 입니다.
* **GPU(T)** : T는 Training을 가르키며, GPU기반 학습할때 초당 1 Batch 데이터를 처리하는 속도입니다.
* **GPU(P)** : P는 Predict를 가르키며, GPU에서 학습이 완료된 모델을 갖고서 1 Batch 데이터를 추론/예측 하는 속도 입니다.