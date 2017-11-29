---
layout: post
title:  "Humoungous - Basic Statistics"
date:   2014-02-03 01:00:00
categories: "statistics"
asset_path: /assets/posts/Humoungous-Statistics/
tags: ['linear regression', 'STD', 'Variance', 'Standard Deviation', 'Conditional Probability']

---

# Linear Regression

$$ slope = \frac{cov(x, y)}{var(x)} = \frac{ \sum{(x- \bar{x}}) ( y - \bar{y}) }{ \sum{ (x - \bar{x})^2 } } $$ 

$$ y \ intercept = \bar{y} - slope * \bar{x}  $$

**R**

{% highlight r %}
lm(y~x)

Coefficients:
(Intercept)            x  
     15.500        2.357  
{% endhighlight %}

**Numpy**

{% highlight python %}
A = np.vstack([x, np.ones(x.size)]).T
slope, y_intercept = np.linalg.lstsq(A, data)[0]
#  slope: 2.357143 
#  y-intercept: 15.500000
{% endhighlight %}

**Pandas**

{% highlight python %}
x = pd.Series(x)
data = pd.Series(data)

slope = x.cov(data) / x.var()
y-intercept = data.mean() - slope * x.mean()
{% endhighlight %}

### Pearson Correlation

* 일반적으로 correlation이라함은 Pearson Correlation을 말한다.
* -1 그리고 1사이의 범위를 갖는다.
* 극으로 갈수록 서로간의 Linear Relationship이 강하며, 0이라면 상관관계가 전혀 없다.
* 공식의 $$ \rho $$ 는 소문자 p와는 다른, rho 기호로서 Person's Correlation을 가르킨다. 
* 공식의 $$ \sigma $$ sigma 기호는 Standard Deviation을 가르킨다.
* 음수가 나오면 X값이 커질수록 Y값의 감소를 의미한다.
* 0.1 ~ 0.3 정도면 상관관계가 약한편이다. (음수도 동일)
* 0.3 ~ 0.5 사이면 적당히 상관관계가 있다. (음수도 동일)
* 0.5 이상이라면 상관관계가 강한 편이다. (음수도 동일)

$$ \rho_{x, y} = Corr(x, y) = \frac{Cov(x, y)}{\sigma_{x} \sigma_{y}} $$

**R**

{% highlight r %}
cov(data$temperature, data$distress_ct)/(sd(data$temperature) * sd(data$distress_ct))
{% endhighlight %}

**Python**

{% highlight python %}
x_mean = np.mean(xs)
y_mean = np.mean(ys)
cov = np.sum((xs - x_mean) * (ys - y_mean))/len(xs)
return  cov /(np.std(xs) * np.std(ys))
{% endhighlight %}

# Measures of Variation 

### Variance 

population & sample variances

$$ \sigma^{2} = \frac{ \sum{}{(x - \mu)^{2}} }{ N } \ , \ s^2 = \frac{ \sum{}{ (x - \bar{x})^{2} } }{ n- 1} $$

{% highlight python %}
np.var(data) # Population Variance
np.var(data, ddof=1) # Sample Variance
{% endhighlight %}

### Standard Deviation
* Standard Deviation은 평균값에서 평균적으로 얼마나 값들이 떨어져 있는지를 나타냅니다.<br>
* Variance에다가 Root만 씌워주면 됩니다. 즉.. $$ \sigma = \sqrt{ variance } $$. <br>
* 모든 값이 같다면 STD는 0
* STD는 평균값에 크게 영향을 받습니다.

{% highlight python %}
np.std(data) # Population Standard Deviation
np.std(data, ddof=1) # Sample Standard Deviation
{% endhighlight %}

### Coefficient of Variation (CV) 변동계수

$$ CV = \frac{\sigma}{\mu}100\% = \frac{STD}{mean}100\% $$

서로 다른 측정단위 또는 방법을 사용하는 데이터를 비교하고자 할때 사용합니다.<br>
예를 들어서 우유공장에서 Quality Inspector가 작은 우유병, 큰 우유병을 비교하고자 합니다.

| 타입 | 평균 우유 양 | STD | ETC | CV |
|:----|:-----------|:-----|:---|:---|
| 작은 우유병 | 1 cup | 0.08 |  | $$ \frac{0.08}{1} * 100 = 8 $$|
| 큰 우유병 | 16 cups | 0.4 | STD가 5배는 높아 보임 | $$ \frac{0.4}{16} * 100 = 2.5 $$ |

즉 큰 우유병이 STD로 볼때 더 편차가 심해보이지만, <br>
두 데이터를 CV로 비교해봤을때 작은 우유병이 평균값에서 상대적으로 더 큰 편차를 보인다는 뜻입니다. 


### Chebyshev's Theorem

데이터가 어떻게 생겼든 상관없이, 최소 $$ \begin{align} (1- \frac{1}{k^{2}}) * 100\% \end{align} $$
안에 데이터가 포함이 된다는 이론입니다. (k는 1보다 큰 값)

| K | Value | Explanation |
| 2 | 75% | 최소 75%의 observations은 2 std 안에 포함됨 (75% 이상) |
| 3 | 89% | 최소 89%의 observations은 3 std 안에 포함됨 (89% 이상) |
| 4 | 94% | 최소 94%의 observations은 4 std 안에 포함됨 (94% 이상) |

예를 들어서 평균 300,000$ 그리고 STD 50,000$의 주택 가격 데이터가 있다.<br>
$$ \begin{align} 1 - \frac{1}{k^{2}} \end{align} = 0.75 $$ 즉 k=2. 
75%의 범위(range)는 $$ \mu  - k\sigma $$ 와 $$ \mu + k\sigma $$ 의 사이인 200,000 ~ 400,000가 됩니다.

{% highlight python %}
# 2 STD 안에 들어가는 Observations의 갯수
m = np.mean(data)
s = np.std(data)
np.sum(np.logical_and(data >= m - 2*s, data <= m + 2*s))
{% endhighlight %}

# Probability


<img src="{{ page.asset_path }}probability.jpg" class="img-responsive img-rounded">

| Term | Definition | Example |
|:-----|:-----------|:--------|
| Mutually Exclusive Events | 동시에 일어나지 못하는 이벤트<br>하나의 class에만 속함(중복X) | 동전앞면 나오기 (뒷면도 동시에 나올수 없다) |
| Independent Events | 서로 영향을 주지 않는 이벤트<br>P(A\|B) = P(A)| |
| Addition Rule | P(A or B) = P(A) + P(B) - P(A or B) | |
| Multiplication Rule | P(A and B) = P(A)P(B\|A) | |
| Complement Rule | P(A`) = 1 - P(A) | 최소 1개가 나오는 확률. -> 1 - 그 반대의 확률 |


### Conditional Probability

$$ P(A|B) = \frac{P(A \ and \ B)}{P(B)} = \frac{ P(A \cap B)}{P(B)} $$

Short Time Warm-up을 갖었을때 Deb이 이길 확률은?

| Warm-up Time | Deb Wins | Bob Wins | Total |
|:-------------|:---------|:---------|:------|
| Short | 4 | 6 | 10 |
| Long  | 16 | 24 | 40 |
| Total | 20 | 30 | 50 |

$$ P(Deb | Short) = \frac{P(Deb \cap Short)}{P(Short)} = \frac{\frac{4}{50}}{ \frac{10}{50}} = \frac{4}{10} = 0.4 $$

이때 P(Deb \| Long) = P(Deb)이므로 Deb 와 Long은 Independent이다.

두번째 예제...

$$ P(A|B) = \frac{P(A)P(B|A)}{P(B)} $$

|      | Sedan | SUV | Total |
|:-----|:---------|:---------|:------|
| New  | 24 | 15 | 39 |
| Used | 9  | 12 | 21 |
| Total | 33 | 27 | 60 |


세단이라는 조건하에, 랜덤으로 선택했을때 New 가 선택될 확률?

$$ P(New|Sedan) = \frac{P(New)P(Sedan|New)}{P(Sedan)} = \frac{0.65 * 0.6154}{0.55} = 0.727 $$


### Permutation

$$ P = \frac{n!}{(n-r)!} $$

Permutatino은 순서가 중요하다. 예를 들어서 [A, B, C]가 있는데, 이중에 2개의 permutation을 구하고자 한다면, 
AB, AC, BA, BC, CA, CB 즉 6가지가 나오게 된다. (AB != BA)

세일즈맨이 9개의 상점중에 5군데를 방문하려고 합니다. 몇가지 방법으로 방문가능합니까?

$$ P = \frac{9!}{(9-5)!} = \frac{9 * 8 * 7 * 6 * 5 * 4!}{4!} = 9 * 8 * 7 * 6 * 5 = 15120$$

### Combination

Permutation과는 반대로 순서가 중요하지 않습니다.

$$ C = \frac{n!}{(n-r)!r!} $$

>4개의 blue구슬, 6개의 red구슬이 있는데, 3개를 꺼낼때 정확하게 2개의 blue구슬만 꺼낼 확률은? 


2개의 blue구슬을 꺼낼 확률 * 1개의 red 구슬을 꺼내는 확률

$$ \frac{4!}{2!2!} * \frac{6!}{1!1!} = \frac{12}{2} * \frac{6}{1} = 36 $$

36가지의 combinations이 나올수 있고, 36/전체 combinations을 해주면 확률이 나옵니다. 즉 .. 

$$ \frac{36}{ \frac{10 * 9 * 8}{3 * 2 * 1}} = \frac{720}{6} = 120 $$

$$ \frac{36}{120} = 0.30 $$

> 3개를 꺼낼때 **최소** 2개의 blue구슬을 꺼낼 확률은?

P(2개의 blue 구슬 + 1개의 red 구슬) + P(3개의 blue구슬) 해주면 되고 이때 중요한 덤은 Addition을 해준다는 것. 

$$ 0.3 +  \frac{\frac{4*3*2}{3*2*1}}{120} = 0.3 + \frac{4}{120} = 0.3 + 0.0333 = 0.3333 $$












