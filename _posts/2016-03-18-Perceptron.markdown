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

* [Iris Data][iris-data]

1930sus 식물학자 Edgar Anderson은 붓꾳(Iris)에 대한 데이터를 수집했습니다.<br>
그는 그 데이터가 현대 머신러닝뿐만 아니라 데이터 싸이언스의 기초 과정이 되리라고는 전혀 예상치 못했겠죠.<br>
었쨌든 iris 데이터는 수많은 머신러닝의 테스트 케이스또는 기초 연구용으로 사용되는 아주 중요한 데이터입니다.<br>
데이터는 다음과 같이 구성되어 있습니다.

* Sepal.Length
* Sepal.Width
* Petal.Length
* Petal.Width
* Species

{% highlight python %}
df = pd.read_csv('iris.csv', header=None)
setosa = df[df[4] == 'Iris-setosa']
versicolor = df[df[4] == 'Iris-versicolor']
virginica = df[df[4] == 'Iris-virginica']

a, b = 0, 3
plt.scatter(setosa[a], setosa[b], color='red', marker='o', label='setosa')
plt.scatter(versicolor[a], versicolor[b], color='blue', marker='x', label='versicolor')

plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.legend(loc='upper left')
plt.grid()
plt.show()
{% endhighlight %}


<img src="{{ page.static }}setosa_versicolor.png" class="img-responsive img-rounded">


# to Python

Perceptron

{% highlight python %}
import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        """
        :param eta: Learning Rate (between 0.0 and 1.0)
        :param n_iter: Training Dataset.
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit Training Data
        """

        self._w = np.zeros(1 + X.shape[1])
        self._errors = []

        for i in range(self.n_iter):
            errors = 0
            for x, target in zip(X, y):
                update = self.eta * (target - self.predict(x))
                self._w[1:] += update * x
                self._w[0] += update
                errors += int(update != 0.0)

            self._errors.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self._w[1:]) + self._w[0]

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

{% endhighlight %}


{% highlight python %}
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 3]].values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn._errors) + 1), ppn._errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
{% endhighlight %}

Perceptron Machine Learning 을 이용해서 기계에 학습을 시킨후 에러률을 출력해봤습니다.<br>
처음에 에러가 2~3개정도씩 나오다가.. 대략 7번이후부터는 정확하게 Classification을 하는 것을 볼수 있습니다.

<img src="{{ page.static }}errors.png" class="img-responsive img-rounded">


[iris-data]: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data


