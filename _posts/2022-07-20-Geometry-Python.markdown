---
layout: post 
title:  "Basic Trigonometric in Python"
date:   2022-07-20 01:00:00 
categories: "math"
asset_path: /assets/images/ 
tags: ['interview', 'math', 'sin', 'cos', 'tan']
---

최근에 코딩 테스트를 봤는데, arctan 물어보는 것이 나왔습니다. <br> 
다행히 모빌리티에서 geometry 관련 코딩을 해놓은게 있어서 코딩 테스트는..<br>
다행히 풀었지만, 기억을 더듬더듬 했습니다. 그래서.. 다시 정리. 


# 1. Basic Geometry 

## 1.1 Degree and Radian 

Python의 삼각 함수는 모두 radian을 기본값으로 사용합니다. 

$$ \begin{align}
Radian &= Degree \times \frac{\pi}{180} \\
Degree &= Radian \times \frac{180}{\pi} \\
2\pi Radians &= 360^o \\
\pi Radians &= 180^o
\end{align} $$

{% highlight python %}
import math

def to_radian(degree):
    return degree * math.pi / 180

def to_degree(radian):
    return radian * 180 / math.pi
{% endhighlight %}

## 1.2 Basic Trigonometry

$$ \begin{align}
sin\theta &= y / r \\
cos\theta &= x / r \\
tan\theta &= y / x \\
r &= \sqrt{x^2 + y^2}
\end{align} $$

 - y: height (높이)
 - x: base (밑변)
 - r: hypotenuse (빗변)