---
layout: post
title:  "Logistic Regression"
date:   2016-05-07 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Logistic-Regression/
tags: ['Logistic', 'Sigmoid', 'binary', 'partial derivative', 'odds ratio', 'maximum likelihood estimation']

---

<div>
    <img src="{{ page.asset_path }}study.jpg" class="img-responsive img-rounded">
</div>


# Odds & Odds Ratio

<img src="{{ page.asset_path }}odds.jpg" class="img-responsive img-rounded">

### Basic Probability

일단 기본 통계부터 잡고 간다면.. 

####  $$ P = \frac{outcomes \ of \ interest}{all \ possible \ outcomes}  $$

| 동전 앞면 나오기 | $$ \begin{align}  P(앞면) = \frac{1}{2} = 0.5 \end{align} $$ |
| 주사위 던져서 1 또는 2 나오기 |  $$ \begin{align}  P(1 \ or \ 2) = \frac{2}{6} = \frac{1}{3}  = 0.333 \end{align} $$ |
| 다이아몬드 카드 나오기 | $$ \begin{align}  P(diamond) = \frac{13}{52} = \frac{1}{4}  = 0.25 \end{align} $$ |

### Odds 


#### $$ odds = \frac{Happening}{Not \ Happening}  = \frac{p}{1-p} $$

공식의 포인트는 Not Happening이 일어날 확률을 구할 필요가 없습니다.<br>
즉 일어날 확률만 알고, 1 - p 해주면 되니, 일어날 확률만 알면 됩니다.

| 동전 앞면 나오기 | $$ \begin{align}  P(앞면) = \frac{0.5}{0.5} = 1 \ or \ 1:1 \end{align} $$ |
| 주사위 던져서 1 또는 2 나오기 |  $$ \begin{align}  P(1 \ or \ 2) = \frac{\frac{2}{6}}{\frac{4}{6}} = \frac{1}{2} = 0.5 \ or \ 1:2 \end{align} $$ |
| 다이아몬드 카드 나오기 | $$ \begin{align}  P(diamond) = \frac{\frac{13}{52}}{\frac{39}{52}} = \frac{1}{3}  = 0.333 \ or \ 1:3 \end{align} $$ |

1:1 이면 성공/실패가 반반이고, 1:2 이면 실패할 확률이 2배 더 많다라고 말할수 있으며, 
1:3 이면 실패할 확률이 3배 더 많다고 말할수 있습니다. 만약 3:1 이라면 성공할 확률이 3배 더 많다라고 말할수 있겠죠. 


### Odds Ratio 오즈비

두개의 odds 의 비율을 나타냅니다.

#### $$ odds = \frac{odds_{1}}{odds_{0}}  = \frac{ \frac{P_{1}}{1-P_{1}} }{ \frac{P_{0}}{1-P_{0}} } $$

### Example

엔지니어링 학과에 남자는 10명중 7명이 입학하며, 여자는 10명중 3명이 입학을 합니다.

| Gender | Success | odds |
| ------ | ------- | ---- |
| 남자 | $$ \begin{align} \frac{7}{10} \end{align}$$ | $$ \begin{align} \frac{0.7}{0.3} = 2.33333 \end{align}$$ | 
| 여자 | $$ \begin{align} \frac{3}{10} \end{align}$$ | $$ \begin{align} \frac{0.3}{0.7} = 0.42857 \end{align}$$ |

위에 나온 데이터로 odds ratio는 $$ \begin{align} \frac{2.33333}{0.42857} = 5.44 \end{align}$$ 과 같습니다.<br>
남자가 여자보다 5.44배 들어갈수 있는 확률이 더 높습니다.


# Logit Function

공식과 그래프는 다음과 같이 생겼습니다.

* p값이 0이 되면 log에 들어가면 undefined 값이 되기 때문에 0은 안됩니다.<br>
* p값이 1이 되면 denominator가 0값이 되기 때문에, 0으로 나누는 꼴이 됨으로 1도 안됩니다.
* p값이 0.5일때 y값은 0이 됩니다.

<img src="{{ page.asset_path }}logit.png" class="img-responsive img-rounded">

Logit Function을 사용하는 이윤는 2개의 claases를 갖은 Binary Classification에서 Linear Regression을 만들기 위해서 입니다.
즉 $$  \begin{align} \ln{\frac{p}{1-p}} = y \end{align} $$ 처럼 단순히 y값이 아니라
$$  \begin{align} \ln{\frac{p}{1-p}} = \beta + \beta x \end{align} $$ 같은 Linear regression을 찾기 위함입니다.
아래의 그림처럼 2개의 클래스의 상관관계가 Linear Regression 으로 표현이 되었습니다.

<img src="{{ page.asset_path }}binary-graph.png" class="img-responsive img-rounded">


#### <span style="color:red"> $$ logit(p(y=1|x)) = w_{0}x_{0} + w_{1}x_{1} + ... + w_{m}x_{m} = \sum_{i=0} w_{m}x_{m} = w^Tx $$ </span>

x features가 주어졌을때 y=1일 확률을 Maximum Likelihood Estimation을 통해서 알아보면 위의 같은 공식이 나옵니다.

### From Probability -> odds -> Logit
최종적으로 Logistic Regression을 하기 위해서는 다음과 같이 변형해야 합니다.

* 확률 P -> Odds -> Logit (log odds)

{% highlight bash %}
p = np.arange(0.01, 1, 0.05)
odds_data = p/(1-p)
logit_data = np.log(odds_data)
{% endhighlight %}

<img src="{{ page.asset_path }}odds_logit.png" class="img-responsive img-rounded">

이렇게 변환하는 이유는 확률자체가 갖고 있는 restricted range (0~1)의 범위로는 어떠한 Model을 찾기가 매우 어렵기 때문입니다.
확률은 0~1사이의 제한된 범위를 갖고, odds는 0~infinite 의 범위를 갖지만 음수의 범위가 없습니다. 
logit을 하면 그래프에 보이듯이 -infinite ~ infinite 사이의 범위를 갖기 때문에 모델링 하기가 쉬워 집니다.

<img src="{{ page.asset_path }}estimated_regression.png" class="img-responsive img-rounded">



# Logistic Function

Logistic Function은 S자 형태라서 **Sigmoid Function**으로도 불리며, logit function의 **inverse function** 입니다.


<img src="{{ page.asset_path }}logistic.png" class="img-responsive img-rounded">

Logistic Function의 정의는 다음과 같습니다.

#### $$\phi(z) = \frac{1}{1+e^{-z}} = \frac{ e^{z} }{ 1 + e^{z} } $$ 

여기서 z는 **net input** 으로서 위의 logit(p(y=1\|x)) 공식입니다.

Python 에서는 다음과 같이 표현 가능합니다.

{% highlight python %}
def logistic(z): 
    return 1 / (1 + np.exp(-z))
{% endhighlight %}



# Learning the weights

### Sum Squared Error Cost Function

Cost Function은 다음과 같습니다.

$$ J(w) = \sum{ \frac{1}{2}( \phi(z^{i}) - y^{i})^{2}  } $$

* z 는 net input으로서 $$ z = \sum{x_{j} w_{j} } = W^{T}X $$
* $$ \phi{(z^{i})} $$ 는 activation function 으로서 logistic function(or sigmoid function)의 값이다. 
* Learning시에 Cost function의 값은 줄이고, Likelihood는 증가시킨다. 

### Likelihood

Cost function을 미분 (derivative) 하는 방법을 설명하려면 먼저 Likelihood $$ L $$ 을 먼저 정의해야 합니다.

$$ L(w) = P(y | x;w) = \prod^{n}{ P(y^{i} | x^{i}; w) } = \prod^{n} (\phi(z^{i}))^{y^{i}} (1 - \phi(z^{i}))^{1 - y^{i}} $$

<img src="{{ page.asset_path }}maximum_likelihood.png" class="img-responsive img-rounded">


# Partial Derivative

$$ f(x,y) = 3x - 2y^4 $$ 처럼 함수 f가 2개 이상의 inputs을 받는 경우 partial derivative를 사용합니다.
이때 1개는 변수로 생각하며, 다른 나머지 값들은 상수로 여길수 있습니다.
예를 들어 y값이 상수로 고정되어 있고, x가 변하는 경우의 derivative 또는 그 반대 이겠죠.

### Example 1

####  $$ f(x, y) = 3x - 2y^4 $$

Partial derivative를 하면 x는 1로 바뀌고, y는 상수이므로 0값으로 바뀝니다.

<span>
$$ \begin{align} 
\frac{\partial f}{\partial x} = f_{x} = 3
\end{align} $$
</span> 


<span>
$$ \begin{align} 
\frac{\partial f}{\partial y} = f_{y} = -2*4y^3 = -8y^3
\end{align} $$
</span>

### Example 2

####  $$ f(x, y, z) = xy^2z^3 + 3yz $$<


<span>
$$ \begin{align} 
\frac{\partial f}{\partial x} = f_{x} = 1 * y^2z^3 + 0 = y^2z^3 
\end{align} $$
</span>

<span>
$$ \begin{align} 
\frac{\partial f}{\partial y} = f_{y} = 2y * xz^3 + 3z * 1 = 2xyz^3 + 3z
\end{align} $$
</span>

<span>
$$ \begin{align} 
\frac{\partial f}{\partial z} = f_{z} = (3z^2) * xy^2 + 3y * 1 = 3xy^2z^2 + 3y
\end{align} $$
</span>






[MNIST Website]: http://yann.lecun.com/exdb/mnist/
