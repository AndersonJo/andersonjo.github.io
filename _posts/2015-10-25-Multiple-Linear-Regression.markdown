---
layout: post
title:  "Multiple Linear Regression"
date:   2015-10-25 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Multiple-Linear-Regression/
tags: ['matrix inverse']


---

<div>
    <img src="{{ page.asset_path }}medical-expenses.jpg" class="img-responsive img-rounded">
</div>


# Formula 

일단 Multiple Regression은 다음과 같이 생겼습니다.

* $$ \alpha $$: y-intercept 
* $$ \epsilon $$: 그리스어로 epsilon이며, Error를 나타냅니다. 

$$ y = \alpha + \beta_{1}x_{1} + \beta_{2}x_{2} + \ ... \ + + \beta_{i}x_{i} + \epsilon $$

$$ \beta_{1} $$,  $$ \beta_{2} $$  처럼 Coefficients들이 각각의 features들이 붙어있는데, 
이는 각각의 feature(x값)들이 따로따로 y값에 대해서 estimated effect를 갖기 위함입니다. 
(즉 각각의 feature들마다 slope이 있다고 생각하면 됨)

이때 y-intercept값은 다른 Regression Parameters들과 별 차이가 없기 때문에 다음과 같이 $$ x_{0} = 1 $$ 로 잡아서 다음과 같이 쓸수 있습니다.
($$ \beta_{0} $$은 beta-naught라고 발음합니다.) -> 계산을 편하게 하기 위함

$$ y = \beta_{0}x_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} + \ ... \ + + \beta_{i}x_{i} + \epsilon $$

<img src="{{ page.asset_path }}regression.png" class="img-responsive img-rounded">

궁극적인 목표는 Sum of the squared errors를 구했을때 error가 가장적은 $$ \beta $$ (The vector of regression coefficients)값을 찾는 것입니다.

$$ \hat{\beta} = (X^{T}X)^{-1}X^{T}Y $$

* T 는 Transpose를 뜻하고, negative exponent는 matrix inverse를 뜻합니다.

**R**

{% highlight r %}
reg <- function(y, x){
  x <- as.matrix(x)
  x <- cbind(Intercept=1, x) # Intercept 라는 column을 추가시킵니다. 안의 데이터는 모두 1값
  b <- solve(t(x) %*% x) %*% t(x) %*% y # solve 는 inverse of a matrix를 취합니다.
  colnames(b) <- 'estimate'
  print(b)
}

reg(y=challenger$distress_ct, x=challenger[2:4])
                         estimate
Intercept             3.527093383
temperature          -0.051385940
field_check_pressure  0.001757009
flight_num            0.014292843
{% endhighlight %}

**Python**

{% highlight python %}
import numpy as np
from numpy.linalg import inv

challenger = np.genfromtxt('challenger.csv', delimiter=',', skip_header=True)

def reg(x, y):
    x = np.c_[np.ones(len(x)), x]
    b = inv(np.dot(x.T, x))    
    b = np.dot(np.dot(b, x.T), y)
    return b[0], b[1:]
    
y_intercept, coefficients = reg(y=challenger[:, 0], x=challenger[:, 1:4])
# y_intercept:  3.52709338331
# coefficients: [-0.05138594  0.00175701  0.01429284]
{% endhighlight %}


# Matrix Inverse
 
$$ A A^{-1} = I $$

* I 는 Identity Matrix 입니다.

{% highlight python %}
import numpy as np
from numpy.linalg import inv

a = np.array([[4., 7.], [2., 6.]])
# array([[ 4.,  7.],
#        [ 2.,  6.]])

inv(a) 
# array([[ 0.6, -0.7],
#        [-0.2,  0.4]])

np.dot(a, inv(a))
# array([[ 1.,  0.],
#        [ 0.,  1.]])

{% endhighlight %}