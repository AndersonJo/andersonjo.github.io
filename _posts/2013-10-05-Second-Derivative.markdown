---
layout: post
title:  "Second Derivative - Speed, Velocity, Acceleration"
date:   2013-10-05 01:00:00
categories: "calculus"
asset_path: /assets/images/
tags: ['Speed', 'Velocity', 'Acceleration']

---

# Second Derivative

기본적으로 derivative 는 어느 특정 지점(point)에서 특정 함수의 slope 또는 변화량(rate of change)을 계산합니다. <br>
"Second Derivative" 는 해당 함수의 미분된 값을 다시 미분시키는 것으로 보면 됩니다.

수식기호로는 다음과 같이 표현합니다.

1. 함수에 대한 derivative : $$ f^{\prime}(x) $$
2. Second derivative of the derivative of a function : $$ f^{\prime \prime}(x) $$

예를 들어 $$ f(x) $$ 는 다음과 같습니다.

$$ f(x) = x^3 $$

f함수는 위와 같은때 미분값은 다음과 같습니다.

$$ \begin{align}
f^{\prime}(x) &= 3x^2 \\
f^{\prime \prime}(x) &= 6x
\end{align} $$


## Rules

**Power Rule**

$$ \begin{align} \frac{d^2}{dx^2} \left[ x^n \right]
&= \frac{d}{dx} \frac{d}{dx} \left[ x^n \right]   \\
&= \frac{d}{dx} \left[ n x^{n-1} \right] \\
&= n \frac{d}{dx} \left[ x^{n-1} \right] \\
&= n (n-1) x^{n-2}
\end{align} $$





# Distance, Speed, Velocity, and Acceleration

수식은 다음과 같이 표현

* 거리 : $$ s $$
* 속력 : $$ v $$ (velocity의 v일거 같지만 일반적으로 speed를 v로 씀)

##  Speed

**평균 속력(Average Speed)**는 다음과 같습니다.

$$ \begin{align} \bar{v} = \frac{\Delta s}{\Delta t} \end{align} $$


여기 속력에서 $$ \Delta $$ 라는 것 자체가 변화량(change)를 가르킵니다.<br>
Calculus에서 변화량은 derivative를 가르키며 **순간적인 속력(Instantaneous Speed)** 는 다음과 같습니다.

$$ v = \lim_{\Delta t \rightarrow 0} \frac{\Delta s}{\Delta t} = \frac{ds}{dt} $$

즉 정말로 우리가 구하고자 하는 **평균속력 $$ \bar{v} $$ **가 아닌 **속력 $$ v $$ **의 값을 얻을 수 있습니다. <br>
**속력** 는 Calculus에서는 **first derivative of distance with respect to time** 으로 봅니다.<br>
뭐 하지만 여기서 평균속력이나 순간적인 속력이나 결과는 동일합니다.

### Example

*만약 서울에서 대구까지의 거리는 300km 이고 자동차로 이동하는데 4시간이 걸렸다면, 속력은 다음과 같습니다.*

$$ \bar{v} = \frac{\Delta s}{\Delta t} =  \frac{d \text{s}}{d \text{t}} = \frac{300}{4} = 75\ \text{km/h} $$

## Velocity

* **평균 속력(speed)**는 시간에 따른 거리(distance)의 변화율(rate of change) 입니다.
* **평균 속도(velocity)**는 시간에 따른 변위(displacement)의 변화율(rate of change) 입니다.

Calculus에서는 이렇게 해석이 될 수 있습니다.

* **순간적인 속력(instantaneous speed)** 는 first derivative of distance with respect to time
* **순간적인 속도(instantaneous velocity)** 는 first derivative of displacement with respect to time

> <span style="color:#777777">참고사항 <br>
> 거리(distance)와 변위(displacement) 가 다르듯이 속력(speed)와 속도(velocity)는 다릅니다.<br>
> 속력(speed)는 scalar값이고, 속도(velocity)는 vector값으로서 크기와 방향을 모두 갖고 있습니다.<br>
> 또한 속력(speed)는 $$ v $$ 로 표시하고, velocity는 $$ \mathbf{v} $$ (볼드체)로 나타냅니다. </span>

**평균 속도 (Average Velocity)**

$$ \bar{\mathbf{v}} = \frac{\Delta \mathbf{r}}{\Delta t} $$

**순간 속도 (Instantaneous Velocity)**

$$ \mathbf{v} = \lim_{\Delta t \rightarrow 0} \frac{\Delta \mathbf{r}}{\Delta t} = \frac{d \mathbf{r}}{dt} $$


**순간 속력 = | 순간 속도 |**
순간 속력(Instantaneous speed) 는 순간 속도(Instantaneous velocity)값의 크기와 동일합니다.<br>
속력은 얼마나 빠른가를 말해주고, 속도는 어느방향으로 얼마나 빠른가를 말해줍니다.

$$ v = | \mathbf{v} | $$



## Acceleration

Acceleration은 시간에 따른 속도(velocity)의 변화량(rate of change) 라고 할 수 있습니다.<br>
따라서 속도가 아무리 빠른 비행기라도 정확하게 **동일한 속도**로 그리고 **동일한 방향**으로 하늘을 날고 있다면.. acceleration의 값은 0이 됩니다.

**평균 가속도 (Average Acceleration)**

$$ \bar{\mathbf{a}} = \frac{\Delta \mathbf{v}}{\Delta t} = \frac{\mathbf{v} - \mathbf{v}_0}{\Delta t}  $$

* $$ \mathbf{v} $$ : final velocity
* $$ \mathbf{v}_0 $$ : initial velocity

**순간 가속도 (Instantaneous Acceleration)**

$$ \mathbf{a} = \lim_{\Delta t \rightarrow 0} \frac{\Delta \mathbf{v}}{\Delta t} = \frac{d \mathbf{v}}{d t} $$


Acceleration은 derivative of velocity with time 으로 볼 수 있습니다.

$$ \mathbf{a} = \frac{d\mathbf{v}}{dt} = \frac{d}{dt} \frac{d \mathbf{s}}{dt} = \frac{d^2 \mathbf{s}}{dt^2} $$


### 예제

예를 들어서 시작 지점이 20m 이고 마지막 도착 지점이 60m이고 도착지점까지 걸린시간은 5시간일때 velocity는 다음과 같습니다.

$$ \mathbf{v} = \frac{60 - 20}{5} = 8 \ \text{m/s} $$

여기서 Acceleration값은 다음과 같습니다.

$$ \frac{d^2 s}{dt^2} = \frac{8}{5} = 1.6 \ \text{m}/\text{s}^2 $$