---
layout: post
title:  "TensorFlow - Softmax Regression"
date:   2016-05-05 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/TensorFlow-Softmax-Regression/
tags: ['MNIST', 'hand-written digits', '손글씨']

---

<div>
    <img src="{{ page.asset_path }}digits.png" class="img-responsive img-rounded">
</div>

# MNIST

### MNIST Dataset

[MNIST Website][MNIST Website]에서 MNIST 데이터를 다운받을수 있습니다.

다운로드된 파일은 다음과 같이 구성되어 있습니다.

* **train-images-idx3-ubyte.gz**:  training set images (9912422 bytes) 
* **train-labels-idx1-ubyte.gz**:  training set labels (28881 bytes) 
* **t10k-images-idx3-ubyte.gz**:   test set images (1648877 bytes) 
* **t10k-labels-idx1-ubyte.gz**:   test set labels (4542 bytes)

파이썬으로 집접 보기 위해서는 gzip, struct 를 사용해서 풀면됩니다. 

### MNIST In TensorFlow

MNIST는 [MNIST Website][MNIST Website]에서 다운 받을수 있지만, <br> 
TensorFlow는 이미 자동으로 다운받을수 있는 코드를 포함하고 있습니다.


{% highlight python %}
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
{% endhighlight %}


다운로드된 데이터는 3가지로 분류될수 있습니다.

* **mnist.train** 55,000개의 트레이닝 데이터 
* **mnist.test** 10,000개의 테스트 데이터 
* **mnist.validation** 5,000개의 검증 데이터 

 이미지 샘플을 보기 위해서는 다음과 같은 코드를 하면 됩니다.

{% highlight python %}
fig, subplots = pylab.subplots(8, 10) # subplots(y축, x축 갯수)

idx = 0
for _subs in subplots:
    for subplot in _subs:
        d = mnist.train.images[idx]
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        subplot.imshow(d.reshape(28, 28), cmap=cm.gray_r)
        idx += 1
{% endhighlight %}
 
<img src="{{ page.asset_path }}sample.png" class="img-responsive img-rounded">

[MNIST Website]: http://yann.lecun.com/exdb/mnist/