---
layout: post 
title:  "Basic Geometry in Python"
date:   2022-07-20 01:00:00 
categories: "math"
asset_path: /assets/images/ 
tags: ['interview', 'math', 'sin', 'cos', 'tan']
---

# 1. Basic Geometry 

## 1.1 Degree and Radian 

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