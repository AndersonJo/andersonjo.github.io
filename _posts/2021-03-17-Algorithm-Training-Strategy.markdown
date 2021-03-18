---
layout: post
title:  "Algorithm Trading Strategy"
date:   2021-03-17 01:00:00
categories: "finance"
asset_path: /assets/images/
tags: ['moving-average', 'exponential-ma', 'indicator']
---


# 1. Basic Indicators

## 1.1 Exponential Moving Average

$$ \begin{align} 
EMA &= v_t \times k + EMA(v_{t-1}) \times (1-k)  \\
\text{or} & \\ 
EMA &= (v_t - EMA(v_{t-1})) \times k + EMA(v_{t-1})
\end{align} $$

 - $$ v_t $$ : today
 - $$ v_{t-1} $$ : yesterday 
 - N : number of days over which to average
 - k : 2 / (N + 1)

{% highlight python %}
random.seed(0)
data = [random.randint(0, 20) for i in range(30)]

def cal_ema1(data, n=10):
    k = 2 / (n + 1)

    ema_p = data[0]
    ema = [data[0]]
    for v in data[1:]:
        ema_p = (v - ema_p) * k + ema_p
        ema.append(ema_p)
    return np.array(ema).round(2)
{% endhighlight %}

**Python 1**
{% highlight python %}
def cal_ema2(data, n=10):
    k = 2 / (n + 1)
    
    ema_p = data[0]
    ema = [data[0]]
    for v in data[1:]:
        ema_p = v * k + ema_p * (1 - k)
        ema.append(ema_p)
    return np.array(ema).round(2)
{% endhighlight %}

**Python 2**
{% highlight python %}
def cal_ema2(data, n=10):
    k = 2 / (n + 1)
    
    ema_p = data[0]
    ema = [data[0]]
    for v in data[1:]:
        ema_p = v * k + ema_p * (1 - k)
        ema.append(ema_p)
    return np.array(ema).round(2)
{% endhighlight %}

**Pandas**
{% highlight python %}
ema3 = pd.Series(data).ewm(span=10, adjust=False).mean().round(2).tolist()
{% endhighlight %}