---
layout: post
title:  "TensorFlow - Convolutional Neural Networks"
date:   2016-06-14 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Convolutional-Neural-Networks/
tags: []

---

<header>
    <img src="{{ page.asset_path }}4ZSWD4L-e1436336191130.jpg" class="img-responsive img-rounded" style="width:100%">
</header>

# Convolutional Neural Networks

* Convolution neural network (or ConvNets)은 기존의 뉴럴 네트워크와는 다른 레이어와 방식을 갖고 있습니다.
* Input Data가 Image여야 한다는 제약 조건이 있습니다.
* ConvNets은 기존 뉴럴넷의 경우 이미지 크기가 커지면 Scale이 안되는 단점을 극복합니다.
* 3 dimensions (width, height, depth - colors) 로 구성 되어 있음.
* CIFAR-10 이미지는 32 * 32 * 3 (width, height, depth 각각) 로 구성되어 있습니다.
* CIFAR-10의 마지막 레이어는 1 * 1 * 10으로서 class scores를 갖고 있습니다. 

### Layers

* Convolutional Layer, Pooling Layer, Fully-Connected Layer 로 구성되어 있습니다. 
* CIFAR-10의 경우  [INPUT - CONV - RELU - POOL - FC] 로 구성되어 있습니다.
* **Conv Layer**는 input 데이터의 local regions에 연결된 뉴런들을 연산합니다. <br>12필터를 적용한다면, 32 * 32 * 12의 volume 결과치를 냅니다. (Input Data는 32 * 32 * 3) 
* **RELU Layer**는 elementwise activation function - max(0, x) - 을 적용합니다. <br>0 값에서 thresholding 하며 32 * 32 * 12의 결과치를 냅니다. 
* **POOL Layer**는 downsampling operation을 (width, height)에 적용하며, 16 * 16 * 12 결과치를 냅니다.
* **FC (fully-connected) Layer**는 class scores를 연산하며, 1 * 1 * 10의 결과치를 냅니다.

<img src="{{ page.asset_path }}cnn.jpeg" class="img-responsive img-rounded">


# MNIST Example with TensorFlow


  





[MNIST Website]: http://yann.lecun.com/exdb/mnist/