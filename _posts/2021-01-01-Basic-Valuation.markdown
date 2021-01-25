---
layout: post
title:  "Basic Valuation - PER, PEG, EPS, ..."
date:   2021-01-01 01:00:00
categories: "finance"
asset_path: /assets/images/
tags: ['per', 'eps', 'fundamental-analysis']
---

# 1. Introduction

 - Absolute Valuation
    - 펀더멘탈에 기초한 절대적 가치를 찾음
    - Models 
       - Dividend Discount Model 
       - Discounted Cash Flow Model
       - Resdual Income Model
       - Asset-based Model
 - Relative Valuation
    - 다른 기업과 비교하여 상대적인 가치를 찾음
    - 계산하기 쉬움
    - Models
       - P/E Ratio
       - Current Ratio
       - Earnings per Share
    
 - 분석방법의 차이
    - Quantitative: 주로 재무제표를 의미하며, 수익, 매출, 비용, 자산등에 기반한 분석
    - Qualitative: 정확하게 숫자로 치환하기 어려운 것들.. 브랜드 네임, 특허, 기술 등등..<br> 
      (예. 테슬라가 펀더멘탈에 비해서 주가가 미친듯이 높은 이유)
      
    


# 2. Basic Valuations

## 2.1 유동비율 Current Ratio 

$$ \text{유동비율 (Current Ratio)} = \frac{ \text{유동자산 (Current Assets)}}{\text{유동부채 (Current Liabilities)}} $$
 
 - **운전자본비율 (Working Capital Ratio)** 라고도 함
 - 단기 부채를 커버 할 수 있는지 확인 가능
 - 산업마다 다르지만 2 이상 값이 되면 좋음
 
## 2.2 당좌비율 Quick Ratio 

$$ \text{당좌비율 (Quick Ratio)} = \frac{당좌자산(Quick Assets)}{유동부채(Current Liabilities)} $$

 - **산성시험비율 (Acid Test Ratio)** 라고도 함 
 - 재고자산을 제외하고, 현금및 현금등가물과 매출채권만으로 유동부채를 커버 할 수 있는지 체크
 - 업종에 따라 재고자산의 현금화가 느릴경우 그리고 재고자산이 모두다 제 가격을 받고 현금화가 될 수 있는게 아니기 때문에.. 엄격하게 평가할 때 사용
 - 1 이하면 유동성 위기가 올 수 있음
 - 경기위축때는 유동비율 보다는 당좌비율을 체크! -> 기업 안전성 체크
 
## 2.3 주당 순이익 (Earnings per Share, EPS)

$$ \text{Basic EPS} = \frac{\text{당기순이익(Net Income) - 우선주 배당금(Preferred Dividends)}}{\text{보통주 가중 평균 주식수 (Weighted Average Common Shares Outstanding)}} $$

 - 투자의 관점에서 투자금 대비 얼마나 효율적으로 수익을 벌고 있는지를 체크
 - **당기순이익**: 기업이 **1년 동안** 벌어들은 총수익 (매출액, 영업외수익) - 총비용(매출원가, 판매비와관리비, 영업외비용, 법인세비용)


가중 평균 주식수 라고 되어 있는데 예를 들면 다음과 같습니다. <br> 
기존 발행된 보통 주식수가 1000주가 있고, 7월 1일에 1000주의 신주를 발행했다면 다음과 같이 계산이 됩니다. 

|           | Shares Outstanding | Period Covered | Weighted Shares   | 
|:----------|:-------------------|:---------------|:------------------|
| 1월 ~ 6월  | 1000               | 0.5 = 6/12     | 500 = 1000 * 0.5  |
| 7월 ~ 12월 | 2000               | 0.5 = 6/12     | 1000 = 2000 * 0.5 |
| sum       |                    |                | 1500              |

따라서 최종 weighted shares 는 1500주가 됩니다.<br>

투자자 입장에서 weighted shares 로 계산하는건 별로 좋지 않습니다. <br> 
기업에서 12월에 자사주 매각과 같은 일로 유통주식수를 풀어버리면 EPS는 소폭만 하락합니다. <br> 
투자자 입장에서는 과거보다는 현재가 더 중요합니다. <br> 
따라서 weighted shares를 하는 것 보다는 그냥 현재 주식수로 나눠주는게 더 좋을 수 있습니다.

## 2.4 희석 주당순이익 (Diluted EPS)

$$ \text{Diluted EPS} = \frac{\text{당기순이익} - \text{우선주 배당금} + \text{조정액}}{\text{보통주 주식 수} + \text{잠재적 보통주 주식 수}} $$

 - 희석이란 보통주로 전환될 가능성이 있는 증권을 의미. (따라서 더 보수적으로 볼 수 있음)
 - 전환사채(CB), 신주인수권부 사채, 전환우선주, 스톡옵션 등등..
 

## 2.5 주가수익비율 (Price Earning Ratio, PER)

$$ \text{PER} = \frac{현재 주가}{\text{EPS}} $$

 - 주가가 주당순이익의 몇 배인지를 나타내는 지표
 - 간단하게 PER가 10이면.. 10년이면 투자비용을 모두 회수 할수 있다는 뜻. 
 
 
## 2.6 주가이익증가율 (Price Earnings to Growth Ratio, PEG)

$$ \text{PEG} = \frac{\text{PER}}{\text{예상이익증가율(Earning Growth Ratio)}} $$

 - 예상 이익 증가율은 `EPS증가율` 이며 x 100 해서 퍼센트를 제거한다
 
같은 반도체산업군 안에서 2개의 회사의 PER는 18배, 그리고 22배가 있다고 가정할때, <br>
언뜻 보기에는 22배가 매우 비싸 보이는데 예상 성장률까지 생각한다면 다음과 같이 공식이 바뀔수 있다.<br>
(company1의 성장률은 12% 그리고 company2는 16% 라고 가정)

$$ \begin{align} 
\text{PEG Ratio (company 1)} &= \frac{18}{0.12 \times 100}  = 1.5 \\
\text{PEG Ratio (company 2)} &= \frac{22}{0.16 \times 100}  = 1.375
\end{align} $$

따라서 PER만 보면 company 2가 22배로 더 높게 보이나, 예상 실적 까지 고려를 한다면 더 싸다고 판단 할 수 있다. 


## 2.7 부채비율 (Debt to Equity Ratio, D/E Ratio)

$$ \text{D/E Ratio} = \frac{\text{총부채 (Total Liabilities)}}{\text{자기자본 (Total Shareholders' Equity)}} \times 100\% $$

 - D/E Ratio 라고도 함
 - 네이버에서는 `부채총계/자본총계` 하면 됨. 
 - 일반적으로 100% 이하를 `표준비율` 이라고 하며.. 100% 미만이라는 뜻은 위기시에 기업의 자기자본 금액으로 전체 채무를 해결 할 수 있다는 뜻


## 2.8 자기자본이익률 (Return On Equity, ROE)

$$ \text{ROE} = \frac{\text{당기순이익 (Net Income)}}{\text{자본총액 (Average Shareholders' Equity)}} $$ 


## 2.9 주가순자산비율 (Price to Book Ratio, PBR)

$$ \begin{align} 
\text{순자산가치 (Book Value)} &= \text{자산(Assets)} - \text{부채(Liabilities)} \\
\text{주당 순자산 가치(Book Value per Share)} &= \frac{\text{Book Value}}{주식 수(Outstanding Shares)} \\
\text{P/B Ratio} &= \frac{\text{주가(Market Price per share)}}{\text{Book Value per Share}}
\end{align} $$ 

 - 순자산 가치는 위의 공식에서 알 수 있듯이 총자산에서 부채를 제외한 `자본액`을 의미
 - PBR은 이런 자본액 대비해서 주가가 몇배로 거래되고 있는지 나타냄

