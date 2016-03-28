---
layout: post
title:  "Perceptron"
date:   2016-03-18 01:00:00
categories: "analysis"
static: /assets/posts/Perceptron/
tags: ['python', 'data analytics', 'matchin']
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