---
layout: post
title:  "Basic Trigonometric in Python"
date:   2022-07-20 01:00:00
categories: "math"
asset_path: /assets/images/
tags: ['interview', 'math', 'sin', 'cos', 'tan', 'arctan', 'arcsin']
---

최근에 코딩 테스트를 봤는데, arctan 물어보는 것이 나왔습니다. <br>
다행히 모빌리티에서 geometry 관련 코딩을 해놓은게 있어서 코딩 테스트는..<br>
다행히 풀었지만, 기억을 더듬더듬 했습니다. 그래서.. 다시 정리.

# 1. Basic Trigonometry

## 1.1 Degree and Radian

**Python의 삼각 함수는 모두 radian을 기본값으로 사용**합니다.

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

라이러리 사용시 math 사용

{% highlight python %}
> import math
> 
> math.degrees(2 * math.pi)  # 360
> math.radians(180)  # 3.141596...
{% endhighlight %}

## 1.2 Basic Trigonometry

$$ \begin{align}
sin\theta &= y / r \\
cos\theta &= x / r \\
tan\theta &= y / x = sin\theta / cos\theta \\
csc\theta &= 1 / sin\theta \\
sec\theta &= 1 / cos\theta \\
cat\theta &= 1 / tan\theta \\
r &= \sqrt{x^2 + y^2}
\end{align} $$

- y: height (높이)
- x: base (밑변)
- r: hypotenuse (빗변)
- csc: cosecant (코시컨트)
- sec: secant (시컨트)
- cot: cotangent (코탄젠트)

참고.. 중학교때 배운것들..

| $$ \theta $$    | $$ 0^\circ $$ | $$ 30^\circ $$           | $$ 45^\circ $$           | $$ 60^\circ $$           | $$ 90^\circ $$        |
|:----------------|:--------------|:-------------------------|:-------------------------|:-------------------------|:----------------------|
| $$ sin\theta $$ | 0             | 1/2                      | $$ \frac{\sqrt{2}}{2} $$ | $$ \frac{\sqrt{3}}{2} $$ | 1                     |
| $$ cos\theta $$ | 1             | $$ \frac{\sqrt{3}}{2} $$ | $$ \frac{\sqrt{2}}{2} $$ | $$ \frac{1}{2}        $$ | 0                     |
| $$ tan\theta $$ | 0             | $$ \frac{\sqrt{3}}{3} $$ | 1                        | $$ \sqrt{3} $$           | 0                     |
| Radian          |               | $$ \frac{\pi}{6} $$      | $$ \frac{\pi}{4} $$      | $$ \frac{\pi}{3} $$      | $$ \frac{\pi}{2} $$   |


{% highlight python %}
> math.sin(30 * math.pi / 180)  # 0.5
> math.cos(45 * math.pi / 180)  # \sqrt(2)/2
> math.tan(60 * math.pi / 180)  # \sqrt(3)
{% endhighlight %}

좌표계를 통한 radian값 계산은 다음과 같이 합니다.

{% highlight python %}
def sin(x, y):
    r = math.sqrt(y ** 2 + x ** 2)
    return to_radian(y / r)

def cos(x, y):
    r = math.sqrt(y ** 2 + x ** 2)
    return to_radian(x / r)

def tan(x, y):
    return to_radian(y / x)
{% endhighlight %}


## 1.3 arctangent - x, y 좌표를 알때 각도를 알고 싶을때 

만약 좌표를 알고 있을때, $$ \theta $$ 에 해당하는 radian 값을 알고 싶을때 어떻게 계산하면 될까요?

<img src="{{ page.asset_path }}trigonometry-02.png" class="img-responsive img-rounded img-fluid" style="border:1px solid #aaa;">

arctan 를 사용해서 각도를 계산할 수 있습니다.<br>
arctan는 tan의 역함수이며, 수식으로는 $$ tan^{-1} = arctan $$ 이렇게 사용합니다.<br>
(주의할점은 $$ tan^{-1} $$ 에서 -1의 의미가 역함수라는 뜻이지, -1 지수가 아닙니다. 즉 $$ tan^{-1} \neq  \frac{1}{tan} $$ 입니다)

$$ \begin{align} \theta &= tan^{-1}(\frac{hypotenuse}{base}) \\
&= arctan(\frac{r}{x})
\end{align} $$

- 리턴값의 범위: $$ -\pi $$ 부터 $$ \pi $$ 의 radians 입니다. (즉 0 ~ 180도부터 0 ~ -180도 까지의 범위입니다.)

Python에서는 math.atan2(y, x) -> radian 을 사용합니다.  

{% highlight python %}
# 포인트는 atan2 의 범위는 [-pi/2, pi/2] 이기 때문에 
# 180도 이상부터는 (y값이 음수) 부터는 음수의 영역대 입니다. 
# 중요한 이유는 atan2 의 각도를 array와 함께 사용시 180도에서 -> -179 도로 변하게 되니.. 
# 범위를 찾는경우 circular array 를 사용하는 방법으로 문제를 해결해야 합니다.    
> math.atan2( 0,  1)  # 0 
> math.atan2( 1,  1)  # 0.7853 radians // 45도
> math.atan2( 1,  0)  # 1.5707 radians // 90도
> math.atan2( 1, -1)  # 2.3561 radians // 135도
> math.atan2( 0, -1)  # 3.1415 radians // 180도
> math.atan2(-0.00000001, -1)  # -3.1415 radians // -179.99999도
> math.atan2(-1, -1)  # -2.3561 radians // -135도
> math.atan2(-1,  0)  # -1.5707 radians // -90도
> math.atan2(-1,  1)  # -0.7853 radians // -45도
> math.atan2( 0,  1)  # 0
> math.atan2( 0,  0)  # 0
{% endhighlight %}  