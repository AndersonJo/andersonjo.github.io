---
layout: post
title:  "On Chain Rule, Calculus, and Backpropagation"
date:   2012-01-03 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Backpropagation/
tags: ['RNN', 'Mathematics']

---

<img src="{{ page.asset_path }}neural-network.png" class="img-responsive img-rounded" style="width:100%">

# Calculus' Chain Rule

만약 $$  f(x) = e^{sin(x^2)} $$ 라는 공식이 있다면.. 다음과 같이 풀어서 쓸수도 있습니다.

<p style="color:#E40041">
$$ \frac{d{f}}{d{x}} = \frac{df}{du} * \frac{du}{dh} * \frac{dh}{dx} $$
</p>

이때 $$ f(x) = e^x $$, $$ u(x) = sin(x) $$, 그리고 $$ h(x) = x^2 $$ 이며 f(x)를 풀어서 쓰면 다음과 같습니다.

$$ f(u(h(x))) = e^{u(h(x))} $$

$$ \frac{df}{du} = e^{g(h(x))}, \ \ \frac{du}{dh} = cos(h(x)), \ \ \frac{dh}{dx} = 2x $$

>derivative of e^x 는 그냥 e^x <br>
>derivative of sin(x) = cos(x) <br>
>derivative of x^2 = 2x

따라서 다음과 같은 결론이 날 수 있습니다.

<p style="color:#E40041">
$$ \frac{d{f}}{d{x}} = e^{\sin x^2} * \cos x^2 * 2x $$
</p>












