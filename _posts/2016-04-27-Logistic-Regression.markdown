---
layout: post
title:  "Logistic Regression"
date:   2016-04-27 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Logistic-Regression/
tags: ['Logistic', 'Sigmoid', 'binary', 'partial derivative', 'odds ratio', 'maximum likelihood estimation']

---

<div>
    <img src="{{ page.asset_path }}study.jpg" class="img-responsive img-rounded" style="width:100%">
</div>


# Logistic Regression in R

### Data Cleaning Process

* <a href="{{ page.asset_path }}train.csv">train.csv</a>

데이터를 불러온후 missing values를 체크 그리고 unique values가 있는지 sapply() 사용해서 알아 봅니다.

{% highlight r %}
> train.data.raw <- read.csv('train.csv', header = T, na.strings = c(""))
> sapply(train.data.raw, function(x) sum(is.na(x)))
survived  pclass  name  sex   age  sibsp  parch  ticket  fare  cabin embarked 
       0       0     0    0   177      0      0       0     0    687        2 

> sapply(train.data.raw, function(x) length(unique(x)))
survived  pclass  name  sex   age  sibsp  parch  ticket  fare  cabin embarked 
       2       3   891    2    89      7      7     681   248    148        4 
{% endhighlight %}

Amelia Library를 사용하면 missing values를 plot으로 볼 수 있는 missmap() 함수를 제공합니다.

{% highlight r %}
install.packages('Amelia')
library(Amelia)
missmap(train.data.raw, main='Missing Values vs observed')
{% endhighlight %}

<img src="{{ page.asset_path }}missing_values.png" class="img-responsive img-rounded">


{% highlight r %}
> data = subset(train.data.raw, select = c(1, 2, 4, 5, 6, 7, 9, 11))
> head(data)
  survived pclass    sex age sibsp parch    fare embarked
1        0      3   male  22     1     0  7.2500        S
2        1      1 female  38     1     0 71.2833        C
3        1      3 female  26     0     0  7.9250        S
4        1      1 female  35     1     0 53.1000        S
5        0      3   male  35     0     0  8.0500        S
6        0      3   male  NA     0     0  8.4583        Q
{% endhighlight %}

데이터를 보니 cabin에 687개의 missing values가 존재하고 이는 너무 많으니 cabin은 데이터 분석에서 제외하며,< 
Name

### Tackling Missing Values 

일반적으로 Missing values를 처리하는 방법으로는 mean, mode, 또는 median값으로 대체해주면 됩니다.

{% highlight r %}
data$age[is.na(data$age)] <- mean(data$age, na.rm=T)
{% endhighlight %}

또한 일반적으로 category data (string이라고 생각)는 csv.read할때 기본값이 factor 타입입니다.<br>
어떻게 구성되어 있는지 알기 위해서는 contrasts 함수를 사용합니다.

{% highlight r %}
> is.factor(data$sex)
[1] TRUE
> is.factor(data$embarked)
[1] TRUE

> contrasts(data$sex)
       male
female    0
male      1
> contrasts(data$embarked)
  Q S
C 0 0
Q 1 0
S 0 1
{% endhighlight %}

마지막으로 embarked에 있는 missing values 2건을 삭제 시켜줍니다.

{% highlight r %}
> nrow(data)
[1] 891
> data <- data[!is.na(data$embarked),]
> nrow(data)
[1] 889
{% endhighlight %}

### Model Fitting

일단 데이터는 train용과 test용 2개로 나눕니다.
Logistic Regression을 하기 위해서는 glm()함수를 사용합니다.

{% highlight r %}
> train <- data[1:800,]
> test <- data[801:889,]

> model <- glm(formula=survived~., family=binomial(link='logit'), data=train)
Coefficients:
(Intercept)   pclass  sexmale      age   sibsp    parch     fare embarkedQ embarkedS
   5.137627 -1.08715 -2.75681 -0.03726 -0.2929 -0.11657  0.00152 -0.002656 -0.318786 

> summary(model)
Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.6064  -0.5954  -0.4254   0.6220   2.4165  

Coefficients:
             Estimate Std. Error z value Pr(>|z|)    
(Intercept)  5.137627   0.594998   8.635  < 2e-16 ***
pclass      -1.087156   0.151168  -7.192 6.40e-13 ***
sexmale     -2.756819   0.212026 -13.002  < 2e-16 ***
age         -0.037267   0.008195  -4.547 5.43e-06 ***
sibsp       -0.292920   0.114642  -2.555   0.0106 *  
parch       -0.116576   0.128127  -0.910   0.3629    
fare         0.001528   0.002353   0.649   0.5160    
embarkedQ   -0.002656   0.400882  -0.007   0.9947    
embarkedS   -0.318786   0.252960  -1.260   0.2076    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 1065.39  on 799  degrees of freedom
Residual deviance:  709.39  on 791  degrees of freedom
AIC: 727.39

Number of Fisher Scoring iterations: 5
{% endhighlight %}


### Interpreting the result

summary를 보면은 parch, fare, embarked 데이터는 별로 중요하지 않다고 나옵니다.<br>
또한 sex를 봤을때 p-value값이 가장 낮고, 그 의미는 성별에 따라서 생존과 아주 밀접한 관련을 갖고 있다는 것을 보여줍니다.<br>
negative coefficient는 남자일수록 생존확률이 낮다는 것을 보여줍니다.<br>

* p-value는 낮을수록 신뢰도가 좋다. 
* null hypothesis 귀무가설이란 그냥 말이 안된는 가설을 의미한다.
* p-value <= 0.05 일 경우 밀접한 관련성을 갖음. 
* p-value > 0.05 일 경우 null hypothesis임. 

{% highlight r %}
> anova(model, test='Chisq')
Analysis of Deviance Table
Model: binomial, link: logit
Response: survived
Terms added sequentially (first to last)

         Df Deviance Resid. Df Resid. Dev  Pr(>Chi)    
NULL                       799    1065.39              
pclass    1   83.607       798     981.79 < 2.2e-16 ***
sex       1  240.014       797     741.77 < 2.2e-16 ***
age       1   17.495       796     724.28 2.881e-05 ***
sibsp     1   10.842       795     713.43  0.000992 ***
parch     1    0.863       794     712.57  0.352873    
fare      1    0.994       793     711.58  0.318717    
embarked  2    2.187       791     709.39  0.334990    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
{% endhighlight %}

The difference between the null deviance and the residual deviance shows how our model is doing against the null model (a model with only the intercept). The wider this gap, the better. Analyzing the table we can see the drop in deviance when adding each variable one at a time. Again, adding Pclass, Sex and Age significantly reduces the residual deviance. The other variables seem to improve the model less even though SibSp has a low p-value. A large p-value here indicates that the model without the variable explains more or less the same amount of variation. Ultimately what you would like to see is a significant drop in deviance and the AIC.

### Predicting y

{% highlight r %}
> fitted.results <- predict(model, newdata=test, type='response')
> fitted.results <- ifelse(fitted.results > 0.5, 1, 0)
> mean(fitted.results == test$survived)
[1] 0.8426966
{% endhighlight %}

정확도는 0.84정도가 나오면 나쁘지 않은 수치입니다.


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
