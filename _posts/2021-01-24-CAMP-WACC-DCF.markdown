---
layout: post
title:  "CAPM, WACC and DCF"
date:   2021-01-23 01:00:00
categories: "finance"
asset_path: /assets/images/
tags: ['fundamental-analysis', 'DCF']
---

# 1. Basic   

## 1.1 Basic 용어정리

### 1.1.1 영업이익 (Operating Income)

$$ \begin{align}
 \text{Operating Income} &= \text{Total Revenue} - \text{} 
&= 매출액 - 매출원가 - 영업비용
\end{align} $$ 
    




## 1.1 VIX (CBOE Volatility Index)

VIX 지수는 `시카고 옵션 거래소 변동성 지수 (Chicago Board Options Exchange (CBOE) Volatility Index)` 를 의미합니다.<br>
VIX 는 S&P 500 지수의 옵션 가격에 기초하며, 향후 30일간 지수의 풋옵션 그리고 콜옵션 가중 가격을 결합하여 산정합니다.

보통 `공포 지수`로 잘 알려져 있으며, 주가가 대폭 상승 또는 하락할 것으로 예상하는 투자자는 옵션을 이용해서 해지를 합니다.<br>
따라서 S&P 500지수와 반대로 움직이는 경향이 있으며, 보통 하락 추세에 옵션에 대해서 더 많은 풋옵션을 매입하며, 이때 VIX지수가 오르게 됩니다.

<img src="{{ page.asset_path }}valuation-vix.png" class="img-responsive img-rounded img-fluid center">

 - 30이상이면 높다고 말하며, 20 이하면 낮다고 판단
 - 2000년 ~ 2007년 평균 19.6 에서, 2008년 리먼 브라더스 사태때 89.86 (11월 20일)에 기록함
 - 한국 Kospi에서는 [KSVKOSPI](https://www.investing.com/indices/kospi-volatility) 가 있음



## 1.2 샤프 지수 (Constant Sharpe Ratio)

$$ S_a = \frac{\mathop{\mathbb{E}} \left[ R_a - R_b \right] }{ \sigma_a} $$

 - $$ R_a $$ : 주식같은 위험 자산의 수익률 (예. S&P500 또는 특정 주식)
 - $$ R_b $$ : 무위험 수익률, 국채, 코스피 같은 기준지표. (예. 10년 만기 미국 국채 또는 코스피)
 - $$ \sigma_a $$ : population standard deviation을 해준다. (즉 std(ddof=0))  
 - **기준지표대비 자산의 `초과 수익률`**이며, 수치가 높을수록 수익률이 좋다 (다만 비례해서 리스크도 높다)
 - 샤프 비율 이라고도 한다
 - constant 가 분은 이유는 cov계산시 $$ \sigma_a $$ 만 계산하기 때문. 즉. 시장상황은 constant값으로 봄

보통 해석은 다음과 같이 합니다.

- $$ S_a > 3.0 $$ : 매우 좋음
- $$ S_a > 2.0 $$ : 좋음
- $$ S_a < 1.0 $$ : 그닥..
- $$ S_a < 0 $$ : 손실 ㅎㅎ


예를 들어서 S&P500의 Constant Sharpe Ratio는 다음과 같습니다.

$$ S_a = \frac{\mathbb{E} \left[ \text{S&P500} - \text{10년만기 미국 국채} \right] }{ \sigma(\text{S&P 500})} $$

Python에서는 다음과 같이 합니다.

{% highlight python %}
import numpy as np

# S&P500 년도별 수익률 (percentage 를 100으로 나눠서 소수점으로 만듬)
s = np.array([ 0.228 ,  0.1648,  0.1245, -0.1006,  0.2398,  0.1106, -0.085 ,
               0.0401,  0.1431,  0.1898, -0.1466, -0.2647,  0.372 ,  0.2384,
              -0.0718,  0.0656,  0.1844,  0.3242, -0.0491,  0.2155,  0.2256,
               0.0627,  0.3173,  0.1867,  0.0525,  0.1661,  0.3169, -0.031 ,
               0.3047,  0.0762,  0.1008,  0.0132,  0.3758,  0.2296,  0.3336,
               0.2858,  0.2104, -0.091 , -0.1189, -0.221 ,  0.2868,  0.1088,
               0.0491,  0.1579,  0.0549, -0.37  ,  0.2646,  0.1506,  0.0211,
               0.16  ,  0.3239,  0.1369,  0.0138,  0.1196,  0.2183])


# 미국 10년만기 국채 년도별 수익률 (percentage 를 100으로 나눠서 소수점으로 만듬)
t = np.array([0.04  , 0.0419, 0.0428, 0.0493, 0.0507, 0.0564, 0.0667, 0.0735,
              0.0616, 0.0621, 0.0685, 0.0756, 0.0799, 0.0761, 0.0742, 0.0841,
              0.0943, 0.1143, 0.1392, 0.1301, 0.111 , 0.1246, 0.1062, 0.0767,
              0.0839, 0.0885, 0.0849, 0.0855, 0.0786, 0.0701, 0.0587, 0.0709,
              0.0657, 0.0644, 0.0635, 0.0526, 0.0565, 0.0603, 0.0502, 0.0461,
              0.0401, 0.0427, 0.0429, 0.048 , 0.0463, 0.0366, 0.0326, 0.0322,
              0.0278, 0.018 , 0.0235, 0.0254, 0.0214, 0.0184, 0.0233])

# Sharpe Ratio (안전자산에 대비한 초과 수익률)
sharp_ratio = (s-t).mean()/np.std(s)
# 0.32816403642800934
{% endhighlight %}


## 1.3 리스크 프리미엄 (Equity Risk Premium, ERP)

**위험 프리미엄 Equity Risk Premium (ERP)** 주식같은 위험 자산에 투자할때,<br>
국채나 은행과 같은 무위험(risk-free) 자산에 비해 `투자자에게 요구되는 최소한의 초과 수익률`을 의미합니다.<br>
쉽게 이야기 하면 하이 리스크, 하이 리턴...<br>
(리스크가 이렇게 높은데 들어갈수 있겠니?)

예를 들어 default위험이 있는 회사의 채권은 이자율이 더 높고, 건실한 기업에서 내놓는 채권의 금리는 싼것과 같습니다. <br>


VIX는 일명 `공포 지수`로서 **위험 지수**를 나타내고, 샤프 지수는 **초과 수익률**을 의미합니다.<br>
VIX 공포지수는 주가가 떨어질때 오르고, 반대로 초과 수익률은 시장이 좋아서 주가가 올라갈때 오릅니다. <br>
주가가 올라갈때는 공포지수가 낮추는 효과가 나오고, 주가가 떨어질때는 공포지수가 높아지면서 서로 상쇄 시키는 효과가 있습니다.<br>
극단적인 상황에서 수익률이 0% 라면, 리스크 프리미엄은 0이 됩니다. <br>
반대로 리스크가 없다면 리스크 프리미엄도 없게 됩니다. <br>
즉 **리스크 프리미엄은 리스크를 동한반 수익률을 의미**합니다. <br>
예제로 1년동안의 ERP 추정치는 다음과 같습니다.

$$ \text{ERP} = \mathbb{E} \left[ \text{1년동안의 Sharpe 지수}\right] \times \mathbb{E}\left[\text{지난 1년동안의 VIX} \right] $$

 - 위의 공식은 예제이고 ERP 공식은 변화 가능.. 하나의 예일 뿐
 - 1년을 가정한 이유는 최소 1년동안 투자한후 홀딩하겠다는 뜻
 - 시간은 분석 하려는 목표에 맞춰서 하면 됨
 - Sharpe 지수는 초과수익률이며 vix는 sharpe 지수와 반대로 주가가 떨어질때 올라간다. <br>

아래 그림에서 ERP평균값 이상으로 올라간때는 다음과 같습니다.

 - 닷컴 IT버블
 - 911 테러
 - 서브프라임 모기지 사태


<img src="{{ page.asset_path }}valuation-erp2.png" class="img-responsive img-rounded img-fluid center">


{% highlight python %}

# 과거 1년동안의 분기별 vix 평균값. (분석에 따라서 월로 쓸지 분기별로 쓸지.. 자유롭게)
vix = np.array([12.86, 15.34, 17.35, 10.31])

# 다음분기 S&P500 예측값과, 10년 국채 예측값을 더 넣어준다.
s2 = np.append(s, 0.0172)
t2 = np.append(t, 0.032)
sharpe_ratio = (s2-t2).mean()/np.std(s2)

# ERP는 sharp_ratio를 반영한다
erp = sharpe_ratio * vix.mean() / 100
# 0.045 = 0.3225 * 13.965 / 100
{% endhighlight %}




## 1.4 연방기금금리 (Federal Funds Rate)

 - 연방준비은행 (Federal Reserve Bank)에 예치되어 있는 지급준비금을 은행 상호간에 1일간 (overnight) 빌릴때 적용되는 금리
 - 은행은 일정이상의 돈을 항시 갖고 있어야 하며, 때문에 은행간 여유가 있는곳이 있고 부족해서 빌려야 하는 곳도 생김 -> 이때 빌릴때 적용되는 금리
 - **연방공개시장위원회 (FOMC, Federal Open Market Committee)** 에서 해당 금리를 결정하며, 매 7주간격 연 8회 FOMC가 열린다
 - FOMC는 **연방기금목표금리 (Federal Funds Target Rate)** 정하게 되며, 0%~0.25% 또는 0.25%~0.5% 같이 금리를 설정하게 됨
 - 은행들은 해당 Target Rate안에서 상호간에 초과분/부족분을 대여/대차 거래를 하게됨
 - 이때 항상 target안에서만 하는건 아니고, 서브프라임 모기지 사태때는 은행들이 긴급하게 돈이 필요해져서 target이상으로 금리가 온라간 적도 있음
 - 실제 은행들간에 거래된 금리의 평균을 **연방기금유효금리 (Federal Funds Effective Rate)**라고 함
 - 관련 데이터는 [https://www.federalreserve.gov/monetarypolicy.htm](https://www.federalreserve.gov/monetarypolicy.htm)에서 문서를 봐야 함

실무적으로 들어와서, FOMC가 FOMC statement를 내놓으면 바로 분석에 들어가야 하며,<br>
문서중에서 빠르게 금리를 찾는 방법은 `target range`로 검색하면 됨<br>
아래서 중요한 부분은 federal funds rate를 0% ~ 0.25% 로 설정하겠다는 뜻

<img src="{{ page.asset_path }}valuation-fomc-statement.png" class="img-responsive img-rounded img-fluid center">

그외 FOMC statement와 같이 보면 좋은 macroeconomic 변수들은 다음과 같습니다.<br>
아래의 내용이 중요한 이유는 statement안에 왜 target rate가 어떻게 설정이 됐는지 나오는데.. <br>
그 주요 변수들이 아래의 것들이며, 잘 이해하고 있다면 다음 target rate도 어느정도 유추할 수 있습니다.

1. GDP
2. Unemployment
3. Personal Consuptions Expenditure (PCE) inflation
4. Core inflation

위의 방식이 어렵다면 [Chicago Mercantile Exchange (CME)](https://www.cmegroup.com/trading/interest-rates/stir/30-day-federal-fund_quotes_globex.html) 에서 Futures 거래를 보면 됩니다.<br>
마지막 가격을 보면 되며, 예를 들어 JAN 2021 의  마지막 가격은 99.91 인데.. 100에서 빼주면 됩니다. <br>
즉 100 - 99.91 = 0.09 이며 이며 현재 target rate (0~0.25) 그 사이임을 알 수 있습니다.

## 1.5 Beta

특정 주식, 펀드, 포트폴리오가 전체 시장과 얼마나 유사한 변동성을 갖는지를 나타낸 지표.<br>
Linear 공식이 y = y_intercept + beta * x_mean 인데.. 여기서 b는 slope을 나타내고..<br>
실제로 이 b가 beta값을 나타냅니다. 즉.. **beta == slope** 이다!

 - $$ \beta > 1.0 $$ : 특정 주가가 시장의 변동폭보다 더 심하게 움직이며, 리스크가 높다
 - $$ \beta = 1.0 $$ : 트정 주가가 시장의 변동폭과 동일하다  
 - $$ \beta < 1.0 $$ : 특정 주가가 시장의 변동폭보다 덜 움직이며, 리스크가 적다
 - $$ \beta = 0 $$ : 특정 주가는 시장의 흐름과 상관관계가 없다
 - $$ \beta < 0 $$ : 특정 주가는 시장의 흐름과 반대로 움직인다 (negatively correlated)

따라서 beta 값은 종종 위험보상(risk-reward)를 측정할때 사용되며, 쉽게 말해.. <br>
투자자로서 얼마나 큰 리스크를 감수하고 잠재적 보상을 얻겠는가 임.<br>
참고로 yahoo finance의 경우 **월별 변화량**을 사용하며, market으로 **S&P500** 를 사용함.


$$ \begin{align}
\beta &= \frac{cov(  s,  m)}{\sigma^2( m)} \\
\end{align} $$

 - $$ \beta $$ : 베타 상관계수
 - $$ s $$ : 특정 주식의 일별 변화량 (percentage)
 - $$ m $$ : 시장의 일별 변화량 (percentage)
 - $$ \sigma^2 $$ : variance


조금 더 설명하면 아래와 같습니다.

$$ \begin{align}
cov(s, m) &= \frac{\Sigma^N_{i=1} (s_i - \bar{s}) \times (m_i - \bar{m})}{N-1} \\
cor(s, m) &= \frac{cov(s, m)}{\sigma_s \times \sigma_m} \\
\sigma_m &= std(m) =  \sqrt{ \frac{\Sigma^N_{i=1} (m_i - \bar{m})^2}{N-1}} \\
\sigma_m^2 &= var(m) =  \frac{\Sigma^N_{i=1} (m_i - \bar{m})^2}{N-1}
\end{align} $$

 - cov: covariance 이며, 문제점이 수치가 커질수록 상관관계가 높은데.. normalization이 안된 상태
 - cor: 상관관계 공식이며, cov에 두 변수의 표준편차를 곱해서 normalization 하는데 -> 결론적으로 -1 ~ 1 사이의 값으로 만듬 <br>
   이걸 넣은 이유는 사실 Beta값은 이론적으로 상관계수에서 나온 공식이라서 그러함. [참고](https://ebrary.net/500/business_finance/mathematical_derivation_beta)
 - $$ \sigma $$ : sample standard deviation 이다. 따라서 분모가 N 이 아니라 N-1 이다.

Python 예제는 다음과 같습니다.

{% highlight python %}
import numpy as np

# 페덱스 일별 주가 변동
s = np.array([ 3.052447,  5.224366,  0.101958,  2.502108,  7.811279,  5.185549,
              -6.122374, -2.556716,  2.952812,  0.776698, -8.855169,  8.284153,
              -0.7809  ])

# S&P500 일별 주가 변동
m = np.array([ 0.054649,  1.930289,  2.218817,  2.80826 ,  0.983162,  5.617872,
              -3.894738, -2.688451,  0.27188 ,  2.160835,  0.48424 ,  3.602159,
              3.026322])
{% endhighlight %}


**Numpy**
{% highlight python %}
def cal_beta(m, s):
    cov = ((s - s.mean()) * (m - m.mean())).sum()
    var = np.sum((m - m.mean())**2)
    return cov/var

cal_beta(m, s)
# 1.1401684598427013
{% endhighlight %}

**Numpy - Matrix**
{% highlight python %}
import statsmodels.api as sm

def cal_beta2(m, s):
    """
    :return : [y-intercept, slope]
    """
    m = sm.add_constant(m)
    a = np.linalg.inv(m.T @ m)
    b = m.T @ s
    B = a @ b
    return B

cal_beta2(m, s)
# array([-0.10172452,  1.14016846])  
{% endhighlight %}

**Numpy - Simple Version**

{% highlight python %}
def cal_beta3(m, s):
    y = np.cov(s, m, ddof=0)/np.var(m)
    return y[0][-1]

cal_beta3(m, s)
# 1.1401684598427015
{% endhighlight %}


**Scipy**
{% highlight python %}
from scipy.stats import linregress

linregress(m, s)
# LinregressResult(slope=1.1401684598427015,
#                  intercept=-0.10172451628899193,
#                  rvalue=0.5716942376087407,
#                  pvalue=0.04122636145990893,
#                  stderr=0.4933667245626671)
{% endhighlight %}















# 2. 자본자산 가격결정모형 (CAPM)

$$ \begin{align}
\text{CAPM} &=  \text{risk-free rate} + \beta \times \text{ERP}
\end{align} $$

 - CAPM: Capital Asset Pricing Model
 - risk-free rate: `10년 만기 미국 국채의 1년 평균`
   - 데이터는 [Fred 에서 다운로드](https://fred.stlouisfed.org/series/DGS10#0)
   - 실제 모델링에서는 Linear Regression모델링 또는 여튼 미래시점의 수익률을 넣는 것이 좋다
 - $$ \beta $$ : 베타값. `1.14`
 - ERP: 리스크 프리미엄. sharpe(0.317) * vix(13.965) = 4.428
 - 캐팸~ 이렇게 읽음

## 2.1 Risk Free Rate

먼저 라이브러리를 불러오고, 10년만기 미국 국채의 최근 1년 평균값을 계산합니다.

{% highlight python %}
import numpy as np
import yfinance as yf
from fredapi import Fred

# 미국 10년만기 미국 국채
fred = Fred(API_KEY)
us10 = fred.get_series('DGS10')
us10 = us10.groupby([us10.index.year, us10.index.month]).mean().iloc[-4:]
# 2020  10    0.787143
#       11    0.870000
#       12    0.933636
# 2021  1     1.082500

risk_free_rate = us10.mean() / 100 # 0.0091
{% endhighlight %}


## 2.2 Beta

베타값을 계산 합니다.

{% highlight python %}
# S&P 500 의 1년간의 월별 변동성 평균
snp = yf.Ticker('^GSPC')
snp = snp.history('5y', '1mo')[['Close']].dropna()
snp = snp.groupby([snp.index.year, snp.index.month]).first()
snp = snp.pct_change().dropna()

# Fedex 1년간의 월별 변동성 평균
fdx = yf.Ticker('FDX')
fdx = fdx.history('5y', '1mo')[['Close']].dropna()
fdx = fdx.groupby([fdx.index.year, fdx.index.month]).first()
fdx = fdx.pct_change().dropna()

# 개월수 맞추기
stocks = fdx.join(snp, rsuffix='_snp')
fdx = stocks.iloc[:, 0].values
snp = stocks.iloc[:, 1].values


# Fedex의 Beta값 = 1.9515830745778673
def cal_beta(m, s):
    n = len(m)
    cov = ((s - s.mean()) * (m - m.mean())).sum()
    var = np.sum((m - m.mean())**2)
    return cov/var

beta = cal_beta(snp, fdx)
# 1.3183174059639897
{% endhighlight %}



## 2.3 Risk Premium

$$ \text{Risk Premium} = \text{Sharpe} \times \text{avg}(\text{1 year vix}) $$

리스크 프리미엄은 샤프지수 * VIX 이며, <br>
특정 종목이 아니라, 주식시장 자체에 투자 했을 경우의 리스크 프리미엄을 계산했습니다.<br>
따라서 S&P500 수익율  그리고 미국 10년 만기 국채 수익율 데이터를 사용하며, <br>
제가 배운 책에서는 역사적 모든 데이터를 사용하지만, 개인적으로는 시대가 달라지니 10년치를 사용하겠습니다.

사용하는 데이터

 - 10년 만기 미국 국채 금리 수익률 (10년 데이터)
 - S&P 500 수익률 (10년치 데이터)
 - S&P 500의 Volatility Measure 로서 CBOE Volatility Index 사용 (1년치 데이터)

{% highlight python %}
# 10년 만기 미국 국채 금리 -> 수익률
# API_KEY는 Fred에서 받아서 넣어야 함
fred = Fred(API_KEY)
us10 = fred.get_series('DGS10')
us10 = us10.groupby([us10.index.year, us10.index.month]).mean()
us10 = us10.pct_change().dropna()  # 퍼센트 -> 소수점 으로 바꾸면서 변화량을 계산

# S&P500 주가 -> 수익률
snp = yf.Ticker('^GSPC')
snp = snp.history('10y', '1mo')[['Close']].dropna()
snp = snp.groupby([snp.index.year, snp.index.month]).first()
sharpe = snp.pct_change().dropna()

# index 맞추기
sharpe['us10'] = us10
snp = sharpe.iloc[:, 0].values
us10 = sharpe.iloc[:, 1].values 

# Constant Sharpe Ratio. 0.3854
sharpe = (snp - us10).mean()/snp.std()
{% endhighlight %}

VIX 데이터를 가져오고, 리스크 프리미엄을 계산합니다.

{% highlight python %}
# VIX for S&P500 Volatility Measure  
vix = yf.Ticker('^VIX')
vix = vix.history('1y', '1mo')[['Close']].dropna()
vix = vix.groupby([vix.index.year, vix.index.month]).first()
#                Close
# Date Date           
# 2020 2     40.110001
#      3     53.540001
#      4     34.150002
#      5     27.510000
#      6     30.430000
#      7     24.459999
#      8     26.410000
#      9     26.370001
#      10    38.020000
#      11    20.570000
#      12    22.750000
# 2021 1     37.209999


# Calculate Equity Risk Premium
# 0.1212 = 0.3854 * 31.4508 / 100
risk_premium = sharpe * vix.values.mean() / 100

{% endhighlight %}



## 2.4 CAPM

$$ \begin{align}
\text{CAPM} &=  \text{risk-free rate} + \beta \times \text{ERP} \\
11.6281 &= 0.9175 + 1.3048 \times 12.1764
\end{align} $$

 - risk-free rate 그리고 ERP 값 모두 한번만 계산해두면.. 마치 상수처럼 모든 종목에 다 쓰임.<br>
   즉 개별종목에 영향을 주는건 beta 밖에없음. (확인했음. 정말 이렇게 계산함)
 - **리스크가 존재할때 위험 정도에 따라서 해당 증권의 적절한 기대수익률**을 계산할때 사용
 - 주주의 요구수익률이 `최소한의 무위험 이자율` 에 `리스크`를 한 이자 수익률을 계산한다고 생각하면 됨

{% highlight python %}
# 0.16900644116976196 = 0.0091 + 1.3183 * 0.1212
capm = risk_free_rate + beta * risk_premium
{% endhighlight %}

CAPM 값이 29.46가 나왔다는 뜻은 required return on equity = 29.46% 라는 것이며, <br>
29.46% 수익률이 나와줘야 한다는 뜻이다. (현재 beta값은 작은데.. GME 사태로 인해서 risk premium이 높아진 상태)














# 3. 가중평균자본비용 (Weighted Average Cost of Capital, WACC)

## 3.1 EBIT & Interest Expense 

$$ \begin{align} 
\text{EBIT} &= \text{Revenue} - \text{COGS} - \text{Operating Expenses} \\
\text{EBIT} &= \text{Net Income} + \text{Interest} + \text{Taxes}
\end{align} $$

 - EBIT: Earnings Before Interest and Taxes
    - 법인세 비용과 이자 비용을 공제하기 전 회사의 순이익
    - Operating Income (영업이익) 도 법인세 비용과 이자 비용을 공제하기 전이라 유사하나.. 다르다.
 - COGS: Cost of Goods Sold (매출원가)    
 - Interest Expense
    - 채권 발행을 통하여 자금을 조달시 세금 절감 효과가 있다
    - **손익계산서 상으로 정부에게 납부해야 할 세금 금액 계산 전에, 채권자에게 주어야 할 이자가 차감 되기 때문이다 (Tax Deductible)**
    - EBIT을 계산하기 전에 interest expense를 차감해주고 -> 그 다음에 세금을 먹인다.
    

<img src="{{ page.asset_path }}valuation-xyz.png" class="img-responsive img-rounded img-fluid center">

\* [Source](https://m.blog.naver.com/PostView.nhn?blogId=yutse&logNo=220504091225&proxyReferer=https:%2F%2Fwww.google.com%2F) 에서 가져온 내용 입니다.


   
## 3.2 WACC


$$ \begin{align}
\text{WACC} &= \frac{\text{E}}{\text{E} + \text{D}} \times R_E + \frac{D}{\text{E} + \text{D}} \times R_D \\
R_E &= \text{CAMP} \\
R_D &= \frac{\text{Total Interest Cost} }{\text{Total Debt}} \times (1-\text{Effective Tax Rate})
\end{align} $$


 - E: Equity (자기자본의 시장가치 = `시가총액 = 발행주식수 x 현재주가`)
 - D: Debt (`부채의 시장가치 = 발행채권수 x 채권의 시장 가격`)
 - V: E + D (기업의 시장가치는 부채와 자기자본의 가치를 합한 것. `총자산 = 자기자본 + 총부채`)
 - E/V: 자기자본으로 조달된 자본의 비중 (%)
 - D/V: 부채로 조달된 자본의 비중 (%). `(1 - E/V)` 로 표현 가능  
 - $$ R_E $$: Cost of Equity (자기자본비용 = `CAPM` 사용)
    - 변동성(위험)에 대해 투자자들이 요구하는 수익률 (리스크가 높으니 일단 기본으로 요정도 수익률은 깔고 가자!)
    - CAPM, DDM(Dividend Discount Model), Arbitrage Pricing Model, PMP 등등.. 여러가지 있다
 - $$ R_D $$: Cost of Debt (타인자본조달비용)
    - After-tax cost of debt: 총금융비용/평균이자발생부채 (1 - 실효세율) 적용
    - Pre-tax cost of debt: 실효세율 적용 없음  



자세하게 알아야 하는 사항 

 - D: Debt
    - Short-term borrowings: 단기 차입금
    - Current portion of long-term debt : 장기부채에 대한 단기 상황금 및 이자 (current = 유동성/1년이내)
        - 야후: Current Debt And Capital Lease Obligation
    - Long Term Debt And Capital Lease Obligation
    - 이때 아래의 계정은 제외 된다
        - (야후) 미지급금 및 미지급 비용 Payables And Accrued Expenses
        - (야후) 연금 및 기타 퇴직 후 혜택 플랜 Pension & Other Post Retirement Benefit Plans Current
        - (야후) 직원 혜택 Employee Benefits  
        - 외상 매입 계정 Accounts payable
        - 미지급 비용 (Accrued liabilities)
        - 이연 매출/수익 deferred revenues
    - Yahoo Finance 
        - Current Debt And Capital Lease Obligation 사용
        - Long Term Debt And Capital Lease Obligation 사용
      <img src="{{ page.asset_path }}valuation-yf-bs.png" class="img-responsive img-rounded img-fluid center">
 - 타인자본조달비용 (Cost of Debt)
    - 예를 들어 회사가 1000원을 4% 이자율로 장기대출 그리고 5% 이자율로 2000원의 채권을 사용한다면..<br>
      `Cost of Debt = (1000*0.04 + 2000*0.05)/(1000+2000) = 0.046` <br>
      여기서 유효세율이 30%라고 가정하면, 세후 부채 비용은..<br> 
      `0.046 * (1-0.3) = 0.032`<br>
      따라서 Cost of Debt는 3.2% 가 됩니다.
    - `672000000/((1974000000 + 964000000 + 64147000000 + 16617000000)/2)`  
    - (야후) Interest Expense Non Operating<br>
        - interest expense (이자비용)은 일반적으로 공제가 가능합니다. 따라서 세금 절감된 비용을 고려하여 계산을 합니다. 
        - $$  $$
      <img src="{{ page.asset_path }}valueation-interest-expense.png" class="img-responsive img-rounded img-fluid center">
    - (야후) 총부채의 2년치 평균값<br>
      <img src="{{ page.asset_path }}valuation-debt.png" class="img-responsive img-rounded img-fluid center">

{% highlight python %}
# 자기자본 / (자기자본 + 총부채)
total_capital = 62682000000  # 시가총액 
short_debt = 1974000000  # 단기 부채
long_debt = 34147000000  # 장기 부채
equity_tc = total_capital/(total_capital + short_debt + long_debt) # equity to total capital 0.6344139347995507  

# 타인자본조달비용 Cost of Debt After Tax
interest_expense = 672000000
effective_tax_rate = 0.25 # 유효세율

short_debt_sum = 1974000000 + 964000000  # Current Debt And Capital Lease Obligation 2년치 평균
long_debt_sum = 34147000000 + 16617000000  # Long Term Debt And Capital Lease Obligation 2년치 평균 
debt_avg = (short_debt_sum + long_debt_sum)/2

cost_of_debt_after_tax = interest_expense/debt_avg * (1-effective_tax_rate)  # 0.018770250642434174


# 최종 WACC 0.11408218342417117
wacc = equity_tc * capm + (1-equity_tc) * cost_of_debt_after_tax

{% endhighlight %}

최종적으로 11.4% WACC값을 계산했습니다. 



























# 5. 용어

## 5.1 Book to Yahoo

book : yahoo 형식

- Income Statement
    - Interest expense : Interest Expense
    

- Balance Sheet
    - Cash and equivalents : Cash, Cash Equivalents & Short Term Investments
    - Spare parts, supplies and fuel, less allowances : Inventory
    
- Cash Flow Statement 
    - Net cash provided by operating activities : Operating Cash Flow
    - Net income : Net Income from Continuing Operations
    - Capital expenditures : Capital Expenditure Reported



## 5.1 자산 Assets

- 유동자산 (Current Assets)
     - 당좌 자산 (Quick Assets)
        - 현금및현금등가물(cash and cash equivalent)
        - 단기금융상품(financial instrument)
        - 유가증권(marketable securities)
        - 단기매매증권(trading securities)
        - 매출채권(trade receivable)-대손충당금
        - 단기대여금(short-term loans)-대손충당금
        - 미수금(non-trade receivables)-대손충당금
        - 미수수익(accrued revenues)
        - 선급금(prepaid payments)
        - 선급비용(advance expenses)
     - 재고 자산 (Inventories)
        - 상품(merchandise)
        - 제품(finished goods)
        - 반제품(semi-finished goods)
        - 재공품(work-in progress)
        - 원재료(raw materials)
        - 저장품(supplies)
        - 부산물(by-products)
        - 미착품(goods in transit)
- 고정자산 (Fixed Assets)
     - 투자자산 (Investments)
        - 투자유가증권(investment securities)
        - 지분법적용투자주식(securities under equity method)
        - 매도가능증권(available-for-sale securities)
        - 장기금융상품(long-term financial instruments)
        - 장기대여금(long-term loans)-대손충당금
        - 장기성매출채권(long-term trade receivables)-현재가치할인자금-대손충당금
        - 투자부동산(investment in real estate)
        - 보증금(guarantee deposit)
        - 이연법인세차(deferred income tax assets)
     - 유형자산 (Tangible Assets)
        - 토지(land)
        - 건물(buildings)-감가상각누계액
        - 구축물(structures)-감가상각누계액
        - 기계장치(machinery)-감가상각누계액
        - 선박(ships)-감가상각누계액
        - 차량운반구(vehicles and transportation equipment)-감가상각누계액
        - 비품(office equipment)-감가상각누계액
        - 건설중인자산(construction in-progress)
     - 무형자산 (Intangible Assets)
        - 영업권(goodwill)
        - 산업재산권(intellectual proprietary rights)
        - 광업권(mining rights)
        - 어업권(fishing rights)
        - 차지권(land use rights)
        - 개발비(development costs)


## 5.2 부채 (Liabilities)

 - 유동부채 (Current Liabilities)
    - 매입채무(trade payable)
    - 단기차입금(short-term borrowings)
    - 미지급이자(accrued interest expense)
    - 미지급금(non-trade payables)
    - 선수금(advances from customers)
    - 미지급비용(accrued expense)
    - 선수수익(unearned revenue)
    - 미지급법인세(income taxes payable)
    - 미지급배당금(dividends payable)
    - 유동성장기부채(current portion of long-term debts)
 - 고정부채 (Long-Term Liabilities)
    - 사채(debentures)-사채발행자금
    - 퇴직급여충당금(provision for severance benefits)
    - 장기차입금(long-term borrowings)
    - 장기성매입채무(long-term trade payable)-현재가치할인자금
    - 임대보증금(leasehold deposits received)
    - 이연법인세대(deferred income tax liablilities)

## 5.3 자기자본 (Owner's Equity)

 - 자본금 (Capital Stock)
    - 보통주자본금(common stock)
    - 우선주자본금(preferred stock)
 - 자본잉여금 (Capital Surplus)
    - 자기주식처분이익(gain on sale of treasury stock)
    - 주식발행초과금(paid-in capital in excess of par value)
    - 감자차익(gain on capital reduction)
    - 기타자본잉여금(other capital surplus)
 - 이익잉여금(Retained Earnings) 또는 누적결손금(Accumulated Deficit)
    - 이익준비금(legal reserve)
    - 기업합리화적립금(reserve for business rationalization)
    - 재무구조개선적립금(reserve for financial structure improvement)
    - 처분전이익잉여금(unappropriated retained earnings carried over to subsequent year)
    - 차기이월결손금(undisposed accumulated deficit carried over to subsequent year)
 - 자본조정 (Capital Adjustments)
    - 주식할인발행차금(discount stock issuance)
    - 자기주식(treasury stock)
    - 미교부주식배당금(unissued stock dividends)
    - 투자유가증권평가이익(gain on valuation of investment securities) 또는 손실(loss on-)
    - 지분법적용투자주식평가이익(gain on valuation of securities under equity method) 또는 손실(loss on-)


