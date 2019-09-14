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
| Book-Value per Share (BPS) | 주당순자산가치 | `BPS = 자본총액 / 주식수` |


# 그린블라트의 마법공식이란?

Greenblatt가 만든 공식으로 `우량기업` 그리고 `염가`를 계량화하여 투자하는 방식을 말합니다.<br>
Warren Buffett의 `좋은회사(profitable companies)를 싼 가격에 산다`는 방식을, <br> 
퀀트의 기준으로 체계적인 공식으로 만든것 입니다.

Greenblatt의 분석에 따르면, 마법공식(magic formula)는 매년 30%이상의 수익을 얻게해줬습니다. <br>
캐쥬얼 투자가에게, Greenblatt는 약 20~30개의 마법공식에 의해서 포트폴리오를 구성하며, 1년정도 주식을 갖고 있는 것을 추천하고 있습니다.

## 우량 기업 (Profitable Companies)

Greenblatt는 Buffett이 어떻게 우량기업(profitable companies)를 정의했는지 참고하였고, <br>
Buffett이 생각하는 우량기업이란 ROE가 높은 회사를 의미한다는 것을 알았습니다. 


<div style="color:red">
$$ \text{우량기업} = \text{자기자본이익률 (Return on Equity Capital, 이하 ROE) 이 높은 기업} $$
</div> 


**Greenblatt는 Buffett이 사용한 ROE를 투하자본이익률 (Return on Invested Capital - ROIC)로 해석을 합니다.**

<div style="color:red">
$$ ROIC = NOPAT / IC $$
</div> 

* ROIC: 투하자본이익률 (Return on Invested Capital)
* NOPAT: 세후순영업이익 (Net Operation Profit after Tax)
* IC: 투하자본(Invested Capital) = 순고정자산 (고정자산 + 고정부채) + 순운전자본 (유동자산 - 유동부채)


단순화를 위해서 