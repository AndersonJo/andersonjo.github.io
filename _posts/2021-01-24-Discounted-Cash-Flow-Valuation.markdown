---
layout: post
title:  "Discounted Cash Flow Valuation (DCF)"
date:   2021-01-23 01:00:00
categories: "finance"
asset_path: /assets/images/
tags: ['fundamental-analysis', 'DCF']
---

# 1. DCF Input Values 

## 1.1 VIX (CBOE Volatility Index)

VIX 지수는 `시카고 옵션 거래소 변동성 지수 (Chicago Board Options Exchange (CBOE) Volatility Index)` 를 의미합니다.<br>
VIX 는 S&P 500 지수의 옵션 가격에 기초하며, 향후 30일간 지수의 풋옵션 그리고 콜옵션 가중 가격을 결합하여 산정합니다.

보통 `공포 지수`로 잘 알려져 있으며, 주가가 대폭 상승 또는 하락할 것으로 예상하는 투자자는 옵션을 이용해서 해지를 합니다.<br> 
따라서 S&P 500지수와 반대로 움직이는 경향이 있으며, 보통 하락 추세에 옵션에 대해서 더 많은 풋옵션을 매입하며, 이때 VIX지수가 오르게 됩니다.

<img src="{{ page.asset_path }}valuation-vix.png" class="img-responsive img-rounded img-fluid center">

 - 30이상이면 높다고 말하며, 20 이하면 낮다고 판단
 - 2000년 ~ 2007년 평균 19.6 에서, 2008년 리먼 브라더스 사태때 89.86 (11월 20일)에 기록함


## 1.2 샤프 지수 (Constant Sharpe Ratio)

$$ S_a = \frac{\mathop{\mathbb{E}} \left[ R_a - R_b \right] }{ \sigma_a} $$

 - $$ R_a $$ : 주식같은 위험 자산의 수익률 (예. S&P500 또는 당신의 한국 포트폴리오)
 - $$ R_b $$ : 무위험 수익률, 국채, 코스피 같은 기준지표. (예. 10년 만기 미국 국채 또는 코스피)
 - $$ \mathbb{E} $$ : 평균 때리면 됨  
 - 기준지표대비 자산의 초과 수익률이며,
 - 샤프 비율 이라고도 한다
 - 보통 0.32 정도 함

예를 들어서 S&P500의 Constant Sharpe Ratio는 다음과 같습니다.

$$ S_a = \frac{\mathbb{E} \left[ \text{S&P500} - \text{10년만기 미국 국채} \right] }{ \sigma(\text{S&P 500})} $$

## 1.3  ERP (Equity Risk Premium)

**위험 프리미엄 Equity Risk Premium (ERP)** 주식같은 위험 자산에 투자할때,<br> 
국채나 은행과 같은 무위험(risk-free) 자산의 수익률 보다 더 얻을 수 있는 `초과 수익률`을 의미 합니다.<br>
쉽게 이야기 하면 하이 리스크, 하이 리턴.. ~~그런데.. 여기에 세금을 매기려고 하는 이!!~~

DCF (Discounted Cash Flow) valuation 은 기본적으로 리스트와 투자 수익률을 연결 시켜서 valuation을 합니다. <br>
위의 VIX는 일명 `공포 지수`로서 위험 지수를 나타내고, 샤프 지수는 초과 수익률을 의미합니다.<br> 
이 둘을 서로 곱하면 리스크 * 초과 수익률이 되며 리스크를 연결시킨 ERP 지표가 나오게 됩니다. <br>
예제로 1년동안의 ERP 추정치는 다음과 같습니다.

$$ \text{ERP} = \mathbb{E} \left[ \text{1년동안의 Sharpe 지수}\right] \times \mathbb{E}\left[\text{지난 1년동안의 VIX} \right] $$
 
 - 위의 공식은 예제이고 ERP 공식은 변화 가능.. 하나의 예일 뿐
 - 1년을 가정한 이유는 최소 1년동안 투자한후 홀딩하겠다는 뜻
 - 시간은 분석 하려는 목표에 맞춰서 하면 됨
 - ERP지수는 각종 위기시마다 급증했음

<img src="{{ page.asset_path }}valuation-erp2.png" class="img-responsive img-rounded img-fluid center">


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
투자자로서 얼마나 큰 리스크를 감수하고 잠재적 보상을 얻겠는가 임. 


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

s = np.array([ 3.052447,  5.224366,  0.101958,  2.502108,  7.811279,  5.185549,
              -6.122374, -2.556716,  2.952812,  0.776698, -8.855169,  8.284153,
              -0.7809  ])
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