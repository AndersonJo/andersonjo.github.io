---
layout: post
title:  "Latent Dirichlet Allocation (LDA)"
date:   2020-11-16 01:00:00
categories: "nlp"
asset_path: /assets/images/
tags: []
---
<header>
    <img src="{{ page.asset_path }}lda-books.jpg" class="img-responsive img-rounded img-fluid center rounded">
    <div style="text-align:right">
    <a style="background-color:black;color:white;text-decoration:none;padding:4px 6px;font-family:-apple-system, BlinkMacSystemFont, &quot;San Francisco&quot;, &quot;Helvetica Neue&quot;, Helvetica, Ubuntu, Roboto, Noto, &quot;Segoe UI&quot;, Arial, sans-serif;font-size:12px;font-weight:bold;line-height:1.2;display:inline-block;border-radius:3px" href="https://unsplash.com/photos/sfL_QOnmy00" target="_blank" rel="noopener noreferrer"><span style="display:inline-block;padding:2px 3px"><svg xmlns="http://www.w3.org/2000/svg" style="height:12px;width:auto;position:relative;vertical-align:middle;top:-2px;fill:white" viewBox="0 0 32 32"><title>unsplash-logo</title><path d="M10 9V0h12v9H10zm12 5h10v18H0V14h10v9h12v-9z"></path></svg></span><span style="display:inline-block;padding:2px 3px">Janko Ferlič</span></a>
    </div>
</header>


# 1. LDA Algorithm Explained 

## 1.1 Introduction

토픽 모델링은 다양한 documents로 부터 토픽을 추출해내는 기법이라고 생각하면 됩니다. <br>
다양한 분야에서 사용이 가능하며, 검색엔진, 과거의 읽었던 책에 기반한 책 추천 등등 다양한 방법으로 활용 될 수 있습니다.<br>
그 중에서 가장 쉽게 적용하면서도 퍼포먼스가 좋은 모델이 LDA (Latent Dirichlet Allocation) 입니다.

 - unsupervised learning (labeling이 필요 없음)
 - statistical model (단어들의 확률론적 분포에 의해서 토픽이 결정됨) 
 - bags of words (즉 단어의 순서 상관 X.. 물론 n-gram으로 어느정도는 해볼 수 있지만..)
 - Topic 갯수$$ \mathcal{T} $$는 미리 정한다. (알고리즘이 알아서 정해주는게 아님)
 
 
아래의 예처럼 vector공간속의 단어의 확륙론적 분포에 따라서 어떤 토픽에 관한 내용인지 알아냅니다. 

<img src="{{ page.asset_path }}lda-example.jpg" class="img-responsive img-rounded img-fluid center rounded" style="border: 2px solid #333333">

## 1.2 수식

 * $$ \mathcal{T} $$ : Topic 갯수
 * $$ \mathcal{D} $$ : Document 갯수
 * $$ \mathcal{V} $$ 
 * $$ \mathcal{N} $$ : document 안의 단어의 갯수. ( $$ \mathcal{N}_i $$ 는 document $$ \mathcal{i} $$ 안의 단어의 갯수)
 * $$ \mathcal{w}_{ij} $$ : 특정 document $$ i $$ 안의 특정 $$ j $$ 번째 단어
 * $$ \mathcal{z}_{ij} $$ : 특정 docuemnt $$ i $$ ... 