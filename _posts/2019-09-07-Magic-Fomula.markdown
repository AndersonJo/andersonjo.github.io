---
layout: post
title:  "Quant 101 and Magic Formula Investing"
date:   2019-09-07 01:00:00
categories: "quant"
asset_path: /assets/images/
tags: ['stockmarket', '주식', '퀀트']
---

<header>
    <img src="{{ page.asset_path }}magic-formula-01.jpg" class="img-responsive img-rounded img-fluid">
    <div style="text-align:right;">
    <a style="background-color:black;color:white;text-decoration:none;padding:4px 6px;font-family:-apple-system, BlinkMacSystemFont, &quot;San Francisco&quot;, &quot;Helvetica Neue&quot;, Helvetica, Ubuntu, Roboto, Noto, &quot;Segoe UI&quot;, Arial, sans-serif;font-size:12px;font-weight:bold;line-height:1.2;display:inline-block;border-radius:3px" href="https://unsplash.com/@austindistel?utm_medium=referral&amp;utm_campaign=photographer-credit&amp;utm_content=creditBadge" target="_blank" rel="noopener noreferrer" title="Download free do whatever you want high-resolution photos from Austin Distel"><span style="display:inline-block;padding:2px 3px"><svg xmlns="http://www.w3.org/2000/svg" style="height:12px;width:auto;position:relative;vertical-align:middle;top:-2px;fill:white" viewBox="0 0 32 32"><title>unsplash-logo</title><path d="M10 9V0h12v9H10zm12 5h10v18H0V14h10v9h12v-9z"></path></svg></span><span style="display:inline-block;padding:2px 3px">Austin Distel</span></a> 
    </div>
</header>

# 기본 용어 정리

주식을 처음 입문하는 ML Engineer로서 기초 경제 용어를 정리해봤습니다.

| English | Korean | Description |
|:--------|:-------|:------------|
| Securities Market | 증권시장 | Security (증권)의 형태가 많기 때문에 반드시 복수형태를 사용한다. 보안 시장이 아니다. <br>안전을 뜻하는 security가 증권으로 사용되는 이유는 돈으로 변환할수 있기 때문이다 |
| Stock Market | 주식시장 | |
| Bond Market  | 채권시장 | | 
| Equity, Share, Stock | 주식 | 주식이란 자산에서 부채를 차감한 잔액으로서 `투자한 소유주의 몫`을 의미. <br>반대로 부채는 채권자가 제공하는 자금 |
| Equity Capital | 자기 자본, 투입 자본 | 소유자지분, 주주지분이라고도 하며, 자산중 부채를 제외한 나머지 금액을 의미 <br>(자본금, 자본조정, 기타포괄손익누계액, 이익잉여금 등등으로 이루어져있다) 
| Capital Stock  | 자본금 | `자본금 = 주식발행총수 x 주식액면금액` | 
| Return on Equity Capital (ROE) | 자기자본이익률 | `ROE = 당기순이익 / 자본총액` -> 투자한 자기자본대비 비용을 뺀 순수한 이익률 |
| Return on Invested Capital (ROIC) | 투하자본이익률 | 
| Book-Value per Share (BPS) | 주당순자산가치 | `BPS = 자본총액 / 주식수` |


# 그린블라트의 마법공식이란?

Warren Buffett의 `좋은회사(profitable companies)를 싼 가격에 산다`는 방식을, <br> 
Greenblatt가 퀀트의 기준으로 체계적 공식으로 만든것 입니다.

Greenblatt의 분석에 따르면, 마법공식(magic formula)는 매년 30%이상의 수익을 얻게해줬습니다. <br>
캐쥬얼 투자가에게, Greenblatt는 약 20~30개의 마법공식에 의해서 포트폴리오를 구성하며, 1년단위로 지속적으로 투자할 것을 추천하고 있습니다.

> 최근 마법공식의 수익률은 낮다고 합니다. <br>
> 참고 정도로만 사용하는게 좋다고 판단 되어 집니다.<br>

## 버펫의 좋은 회사 (Profitable Companies)를 퀀트로 정의하기


### Return on Equity Capital - ROE

Greenblatt는 Buffett이 어떻게 우량기업(profitable companies)를 정의했는지 참고하였고, <br>
Buffett이 생각하는 우량기업이란 ROE가 높은 회사를 의미한다는 것을 알았습니다.<br> 
우량기업이란 자기자본이익률 (Return on Equity Capital, 이하 ROE) 이 높은 기업 이라고 정의할수 있으며, ROE는 다음과 같습니다.


<div style="color:red">
$$ ROE = \text{당기순이익} / \text{자본총액} $$
</div> 
 

ROE의 **최대단점**은 분모가 `자본총액`을 사용한다는 것입니다.<br>
즉 자기자본(주주지분)을 활용해 1년간 얼마나 돈을 벌었는지를 판단합니다.<br>
**문제는 부채가 많은 기업에 대해서 ROE만 갖고서 판단을 한다면, 마치 높은 수익을 거두고 있는 것 처럼 왜곡되어 보일수 있습니다.**


### Return on Invested Capital - ROIC

영업자산으로 얼마만큼의 영업이익을 얻었는가를 평가하는 수익성 지수입니다.<br>
ROE가 주주의 돈에 비례해 얼마나 빨리 벌었는지 확일할때 사용한다면, ROIC는 해당 기업의 비즈니스 수익성을 판단할수 있습니다.

<div style="color:red">
$$ ROIC = NOPAT / IC $$
</div> 

* ROIC: 투하자본이익률 (Return on Invested Capital)
* NOPAT: 세후순영업이익 (Net Operation Profit after Tax) = 영업이익 x (1-법인세율)
* IC: 투하자본(Invested Capital) = 순고정자산 (고정자산 + 고정부채) + 순운전자본 (유동자산 - 유동부채)



### Magic Formula ROIC - MF_ROIC

<div style="color:red">
$$ \begin{align} MF_{ROIC} &= EBIT / MF_{IC} \\
MF_{IC} &= \text{순고정자산} + \text{순운전자본}
\end{align} $$
</div> 

* MF_ROIC: 마법공식 자본이익률 (Magic Formula ROIC)
* EBIT: 이자및 법인세 차감 전 이익 (Earnings before Interest and Tax)
* MF_IC: 마법공식 투하자본 (Magic Formula Invested Capital) = 순고정자산 (부동산 + 공장 설비) + 순운전자본



## 버펫의 싼 가격을 퀄트로 정의하기

<div style="color:red">
$$ \text{이익수익률} = EBIT/EV $$
</div> 

* EBIT: 이자및 법인세 차감 전 이익 (Earnings before Interest and Tax)
* EV: 기업가치(Enterprise Value) = 보통주 시가총액 + 우선주 시가총액 + 부채총액 - 여유현금
* 여유현금 = 현금 + 유동자산 - 유동부채

