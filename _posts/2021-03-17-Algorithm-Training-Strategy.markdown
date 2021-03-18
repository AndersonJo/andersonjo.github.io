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

특징은 다음과 같습니다. 

 - 기존 Moving Average보다 좀 더 최신 데이터에 wegith를 주게 됩니다. 
 - Moving Average 와 동일하게 .. 횡보하게 되면 수익률이 쭉쭉 떨어짐. 

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


## 1.2 Absolute Price Oscillator

$$ \text{Absolute Price Oscillator} = \text{EMA}_{fast} - \text{EMA}_{slow} $$

{% highlight python %}
data =  pd.Series([random.randint(-5, 5) for i in range(300)]).cumsum()

ema1 = pd.Series(data).ewm(span=7, adjust=False).mean()
ema2 = pd.Series(data).ewm(span=20, adjust=False).mean()

# 1.3 Absolute Price Oscillator 
apo = ema1 - ema2
{% endhighlight %}


## 1.4 MACD (Moving Average Convergence Divergence)

$$ \begin{align} 
\text{MACD} &= EMA_{Fast} - EMA_{Slow} \\
\text{MACD}_{Signal} &= EMA_{MACD} \\
\text{MACD}_{Histogram} &= MACD - MACD_{Signal}
\end{align} $$

 - 첫번째줄 MACD는 사실상 APO (Absolute Price Oscillator) 와 동일하다
 - 두번째줄 MACD_Signal은 Raw MACD (APO)에 한번더 smoothing factor 를 입힌 것이다. 
 - MACD_Histogram에서 최종적으로 signal을 찾는다. 

{% highlight python %}
data =  pd.Series([random.randint(-5, 5) for i in range(300)]).cumsum()
ema1 = pd.Series(data).ewm(span=6, adjust=False).mean()
ema2 = pd.Series(data).ewm(span=15, adjust=False).mean()

# MACD = APO
macd = ema1 - ema2
macd_signal = macd.ewm(span=15, adjust=False).mean()  # span은 ema_slow 와 동일하게 가져감
macd_histogram = macd - macd_signal
{% endhighlight %}

<img src="{{ page.asset_path }}trading_macd.png" class="img-responsive img-rounded img-fluid center">
