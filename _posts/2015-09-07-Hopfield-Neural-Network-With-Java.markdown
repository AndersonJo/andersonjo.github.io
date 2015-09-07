---
layout: post
title:  "Hopfield Neural Network with Java"
date:   2015-09-07 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Hopfield-Neural-Network/
---
<div>
    <img src="{{ page.asset_path }}robot.jpg" class="img-responsive img-rounded">
</div>

## Preliminary

[Github Code][github-ann]

gradle로 관련 라이브러리들을 쉽게 설치가능합니다.

{% highlight shell %}

gradle build
gradle eclipse

{% endhighlight %}

## Matrix - Apache Math

Matrix 라이브러리로  Jama 그리고 Apache Common Math 도 있는데 필자는 Apache Common Math를 사용하도록 하겠습니다.
기본적으로 다음과 같은 값을 준비합니다.

{% highlight java %}
double[][] dataA = { { 1, 2, 3 }, { 0, 4, 5 } };
double[][] dataB = { { 3, 3, 3 }, { 3, 2, 0 } };

RealMatrix a = MatrixUtils.createRealMatrix(dataA);
RealMatrix b = MatrixUtils.createRealMatrix(dataB);

{% endhighlight %}

#### Addition


[github-ann]: https://github.com/AndersonJo/Neural-Network-Tutorial