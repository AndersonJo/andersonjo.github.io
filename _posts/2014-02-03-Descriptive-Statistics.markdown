---
layout: post
title:  "Humoungous - Descriptive Statistics"
date:   2014-02-03 01:00:00
categories: "statistics"
static: /assets/posts/Descriptive-Statistics/
tags: ['linear regression']

---

# Linear Regression

$$ slope = \frac{cov(x, y)}{var(x)} = \frac{ \sum{(x- \bar{x}}) ( y - \bar{y}) }{ \sum{ (x - \bar{x})^2 } } $$ 

$$ y \ intercept = \bar{y} - slope * \bar{x}  $$

**R**

{% highlight r %}
> lm(y~x)

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