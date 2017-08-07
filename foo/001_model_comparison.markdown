---
layout: post

date:   2017-08-05 01:00:00
categories: "deep-learning"
asset_path: /assets/images/
tags: ['resnet', 'densenet']

---

# Deep Learning 비교 분석

## System

비교를 하는 시스템의 상황은 다음과 같습니다.

| Name | Value |
|:-----|:------|
| CPU  | Intel I7 6700K CPU 4.00GHz |
| Memory | 32GB |
| GPU  | NVIDIA GTX 1080, 8GHz Memory |
| OS   | Ubuntu 16.04 |


## Models

| Model | Data | Loss | Accuracy | Parameters | Memory | CPU(T) | CPU(P) | GPU(T) | GPU(P) | ETC |
|:------|:-----|:-----|:---------|:-----------|:-------|:-------|:-------|:-------|:-------|:----|
| DenseNet | CIFAR-10 | | | 1 conv -> 3 x 24 dense -> fc | 6619MB |

* **Models**
  - DenseNet : 메모리 최적화되지 않은 [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)의 구현된 모델
* **Parameters**
  - convolution : Convolution Layer
  - fc : Fully Connected Layer (또는 Affine, Linear 라고 보면 됨)
* **BATCH** : 모든 모델의 Batch 의 크기는 32를 사용하였습니다.
* **Memory** : GPU의 메모리 사용량을 의미합니다.
* **CPU(T)** : T는 Training을 가르키며, CPU기반 학습할때 초당 1 Batch 데이터를 처리하는 속도입니다.
* **CPU(P)** : P는 Predict를 가르키며, CPU에서 학습이 완료된 모델을 갖고서 1 Batch 데이터를 추론/예측 하는 속도 입니다.
* **GPU(T)** : T는 Training을 가르키며, GPU기반 학습할때 초당 1 Batch 데이터를 처리하는 속도입니다.
* **GPU(P)** : P는 Predict를 가르키며, GPU에서 학습이 완료된 모델을 갖고서 1 Batch 데이터를 추론/예측 하는 속도 입니다.