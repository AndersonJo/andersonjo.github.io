---
layout: post
title:  "Jupyter & Perceptron"
date:   2016-03-18 01:00:00
categories: "machine-learning"
static: /assets/posts/Perceptron/
tags: ['python', 'data analytics', 'jupyter', 'ipython', 'notebook']
---


<img src="{{ page.static }}analytics.jpg" class="img-responsive img-rounded">

# Installation

### Example

예제는 Python Matplotlib의 plot을 보는 예제

<img src="{{ page.static }}pylab.png" class="img-responsive img-rounded">

### Installing Jupyter

쥬피터는 Ipython Notebook에서 더 발전된 버젼으로 Python, R, Scala등의 데이터 분석에 쓰이는 언어들을 선택해서 웹애플리케이션으로
사용이 가능하게 해줍니다.

{% highlight bash %}
sudo pip install jupyter
jupyter notebook
{% endhighlight %}

### 한글 설정

문서의 가장 윗쪽에 다음과 같이 설정합니다.

{% highlight python %}
#-*- coding:utf-8 -*-
%pylab inline
matplotlib.rc('font', family='NanumGothic')
{% endhighlight %}

만약 내가 갖고 있는 모든 폰트들을 열고 싶다면..

{% highlight python %}
import matplotlib.font_manager
print [f.name for f in matplotlib.font_manager.fontManager.ttflist]
{% endhighlight %}


# Perceptron

### Training Machine Learning Algorithms

<img src="{{ page.static }}perceptron.png" class="img-responsive img-rounded">

Perceptron을 이용해서 Binary Classification을 할 수 있습니다.
(0 과 1처럼 2개의 분류로 나뉘는 것)

### Net Input

* x는 input values로서 1차원 vector입니다.<br>
* w는 그에 일치하는 weight vector입니다.
* z는 net input 입니다. (위의 사진에서 inputs -> weights -> sigma 를 지난 부분)
* *Matrix 의 transpose를 사용해서 sum(sigma)를 대체할수 있습니다.*

<img src="{{ page.static }}net_input.png" >

### Activation Function

Activation Function <img src="{{ page.static }}activation0.png">의 output이
Threshold<img src="{{ page.static }}threshold.png" > 보다 더 크다면 1로 예측할수 있고, 아니라면 -1로 예측 할 수 있습니다.

<img src="{{ page.static }}activation_function.png" >

### Perceptron Learning Steps

초기 MCP neuron 과 Rosenblatt's thresholded perceptron model은 매우 간단합니다.

1. weigts 는 0또는 작은 랜덤값들로 초기화
2. 각각의 training sample <img src="{{ page.static }}x_i.png" > 은 먼저 예측값 <img src="{{ page.static }}predicted_y.png" >
을 알아낸 후, weights를 업데이트 해줍니다.

weights에 대한 업데이트 공식은 다음과 같습니다.


<img src="{{ page.static }}update_w.png" >

<img src="{{ page.static }}delta_w.png">의 값은 Perceptron Learning Rule에 의해서 알아낼수 있습니다.

<img src="{{ page.static }}learning_rule.png" >

* 궁극적으로 <img src="{{ page.static }}predicted_y.png" > 이 y값과 일치하면 0이 되기 때문에 weight에 학습 조정은 없습니다.
* <img src="{{ page.static }}eta.png" > Eta는 Learning Rate로서 0~1사이의 값을 갖습니다.
* 만약 예측값 <img src="{{ page.static }}predicted_y.png" >에 오류가 있다면,  <img src="{{ page.static }}delta_w.png">는
음수 또는 양수로 떨어지게 됩니다.
* <img src="{{ page.static }}delta_w.png">의 크기는 x의 값에 비례해서 달라지게 됩니다.


# Iris Data

[iris-data]: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data