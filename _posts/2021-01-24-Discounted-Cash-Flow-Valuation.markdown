---
layout: post
title:  "CAPM & WACC"
date:   2021-01-23 01:00:00
categories: "finance"
asset_path: /assets/images/
tags: ['fundamental-analysis', 'DCF']
---

# 1. Inputs  

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
sharp_ratio = (s2-t2).mean()/np.std(s2)

# ERP는 sharp_ratio를 반영한다
erp = sharp_ratio * vix.mean()
# 4.428063228992274 = 0.31708293798727344 * 13.965
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















# 2. Models

## 2.1 자본자산 가격결정모형 (Capital Asset Pricing Model, CAPM)

$$ \begin{align}
\text{CAPM} &=  \text{risk-free rate} + \beta \times \text{ERP}
\end{align} $$

 - risk-free rate: `10년 만기 미국 국채의 1년 평균`
   - 데이터는 [Fred 에서 다운로드](https://fred.stlouisfed.org/series/DGS10#0)
   - 실제 모델링에서는 Linear Regression모델링 또는 여튼 미래시점의 수익률을 넣는 것이 좋다
 - $$ \beta $$ : 베타값. `1.14` 
 - ERP: 리스크 프리미엄. sharpe(0.317) * vix(13.965) = 4.428
 - 캐팸~ 이렇게 읽음

### 2.1.1 Risk Free Rate

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

risk_free_rate = us10.mean() # 0.9183198051948052
{% endhighlight %}


### 2.1.2 Beta

베타값을 계산 합니다.

{% highlight python %}
# S&P 500 의 1년간의 월별 변동성 평균
snp = yf.Ticker('^GSPC')
snp = snp.history('5y', '1mo')[['Close']].dropna()
snp = snp.groupby([snp.index.year, snp.index.month]).first()
snp = snp.pct_change().dropna()

# Crisper Therapeutics 1년간의 월별 변동성 평균
crsp = yf.Ticker('CRSP')
crsp = crsp.history('5y', '1mo')[['Close']].dropna()
crsp = crsp.groupby([crsp.index.year, crsp.index.month]).first()
crsp = crsp.pct_change().dropna()

# 개월수 맞추기
stocks = crsp.join(snp, rsuffix='_snp')
crsp = stocks.iloc[:, 0].values
snp = stocks.iloc[:, 1].values


# 크리스퍼 테라퓨틱스의 Beta값 = 1.9515830745778673
def cal_beta(m, s):
    n = len(m)
    cov = ((s - s.mean()) * (m - m.mean())).sum() 
    var = np.sum((m - m.mean())**2)
    return cov/var

beta = cal_beta(snp, crsp)
# 2.41862207283265
{% endhighlight %}



### 2.1.3 Risk Premium 

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
fred = Fred(API_KEY)
us10 = fred.get_series('DGS10') 
us10 = us10.groupby([us10.index.year, us10.index.month]).mean()
us10 = us10.pct_change().dropna()

# S&P500 주가 -> 수익률
snp = yf.Ticker('^GSPC')
snp = snp.history('10y', '1mo')[['Close']].dropna()
snp = snp.groupby([snp.index.year, snp.index.month]).first()
sharpe = snp.pct_change().dropna()

# index 맞추기
sharpe['us10'] = us10
snp = sharpe.iloc[:, 0].values
us10 = sharpe.iloc[:, 1].values

# Constant Sharpe Ratio. 0.386908018529043
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
# 0.386908018529043 * 31.794166882832844 = 12.301418109418575
risk_premium = sharpe * vix.values.mean()

{% endhighlight %}



### 2.1.4 CAPM 

$$ \begin{align}
\text{CAPM} &=  \text{risk-free rate} + \beta \times \text{ERP} \\
11.6281 &= 0.9183 + 2.4186 \times 4.428
\end{align} $$

 - risk-free rate 그리고 ERP 값 모두 한번만 계산해두면.. 마치 상수처럼 모든 종목에 다 쓰임.<br>
   즉 개별종목에 영향을 주는건 beta 밖에없음. (확인했음. 정말 이렇게 계산함)
 - **리스크가 존재할때 위험 정도에 따라서 해당 증권의 적절한 기대수익률**을 계산할때 사용   

{% highlight python %}
# 11.628131270734137
capm = risk_free_rate + beta * erp
{% endhighlight %}

CAPM 값이 11.62가 나왔다는 뜻은 required return on equity = 11.62% 라는 것이며, <br>
11.62% 수익률이 나와줘야 한다는 뜻이다. (그만큼 리스크가 높다는 뜻)

