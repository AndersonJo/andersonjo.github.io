---
layout: post
title:  "Basic Stock Indicators"
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

# Absolute Price Oscillator 
apo = ema1 - ema2
{% endhighlight %}


## 1.3 MACD (Moving Average Convergence Divergence)

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

MACD(histogram) 에서 보듯이 0을 기준으로 오르면 사고, 내릴때는 매도 하는 시그널로 사용하면 됨.

<img src="{{ page.asset_path }}trading_macd.png" class="img-responsive img-rounded img-fluid center">


## 1.4 Bollinger Bands

Bollinger bands는 최근 가격의 변동성을 반영하기 때문에 여러가지 상황에 고정적으로 대응하기 보다 좀 더 유연하게 대처할 수 있습니다.<br>
사용 예제로는 해당 upper 또는 lower 를 주가가 더 튀어 나갈때는 잡습니다. 

$$ \begin{align} 
\text{BB}_{upper} = SMA_{n - periods} + \sigma \times \alpha \\
\text{BB}_{lower} = SMA_{n - periods} - \sigma \times \alpha \\
\end{align} $$

 - SMA, EMA 가 됐든.. 아무거나 사용하면 됨
 - $$ \sigma $$ : standard deviation

{% highlight python %}
data =  pd.Series([random.randint(-5, 5) for i in range(50)]).cumsum()
ema = pd.Series(data).ewm(span=12, adjust=False).mean()

# Bollinger Bands
alpha = 2
std = np.sqrt((data - ema)**2/12)
upper = ema + std * alpha
lower = ema - std * alpha
{% endhighlight %}

<img src="{{ page.asset_path }}trading_bbands.png" class="img-responsive img-rounded img-fluid center">


## 1.5 RSI (Relative Strength)

MA 기반의 Indicator와는 좀 다르게, RSI는 가격의 변화, 강도를 담아냅니다. <br>
50% 이상은 up-trend 를 가르키며, 50% 이하면 downtrend 를 의미 합니다.


$$ \begin{align} 
RSI &= 100 - \frac{100}{1-RS} \\
RS &= \frac{AvgU}{AvgD} \\
AvgU &= \frac{\sum \text{all gain in the last N periods}}{N} \\
AvgD &= \frac{\sum \text{all loss in the last N periods}}{N} \\
\end{align} $$

 - AvgU 에서 loss 부분을 0으로 대체
 - AvgD 에서 gain 부분을 0으로 대체

{% highlight python %}
data = pd.Series([random.randint(-1, 1) for i in range(300)]).cumsum()
gain, loss = data.copy(), data.copy()
diff = data.diff(1)
gain[diff <= 0] = 0  
loss[diff >= 0] = 0

gain_mean = gain.rolling(7).mean()
loss_mean = loss.rolling(7).mean()
rsi = 100 - 100 / (1 - (gain_mean / loss_mean))
{% endhighlight %}

<img src="{{ page.asset_path }}trading_rsi.png" class="img-responsive img-rounded img-fluid center">

## 1.6 Momentum

$$ \begin{align} 
\text{Momentum} = Price_t - Price_{t-n}
\end{align} $$

달리는 말에 올라타는 전략.

## 1.7 Average True Indicator (ATR)


$$ \begin{align} 
TR &= \max[(H-L), abs(H-C_p), abs(L-C_p)] \\
ATR &= \frac{1}{n} \sum^n_{i=1} TR_i
\end{align} $$

 - $$ TR_i $$ : True Range.
 - n : 보통 n=14 
 - Stop-loss 를 어디서 걸어야 되는지를 결정하도록 도와줌

의미
 - 가격폭의 평균 값. -> ATR 값이 높다는건 변동성이 높다는 뜻이고 낮다는건 변동성 작음. 
 - 손절 익절 기준으로도 활용됨.
    - 손절: ATR * 3 
    - 익절: ATR * 1.5 

아래 그림을 보면 쉽게 이해가 됨. 방향을 찾는건 아니고, 어제와 비교해서 가장 크게 움직인 크기를 찾아냄.

<img src="{{ page.asset_path }}trading_tr.png" class="img-responsive img-rounded img-fluid center">

{% highlight python %}
import numpy as np
import pandas as pd

def atr(df, n=14):
    c_p = df.close.shift()
    df['H-L'] = df.high - df.low
    df['H-Cp'] = (df.high - c_p).abs()
    df['L-Cp'] = (df.low - c_p).abs()
    df['TR'] = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    df['ATR'] = df.TR.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    
atr(df)
{% endhighlight %}

<img src="{{ page.asset_path }}trading_tr2.png" class="img-responsive img-rounded img-fluid center">

ZetaValue 내부 툴로 확인한 ATR. (n=14)

<img src="{{ page.asset_path }}trading_atr.png" class="img-responsive img-rounded img-fluid center">

