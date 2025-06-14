---
layout: post
title:  "Sentence Bert"
date:   2025-04-02 01:00:00
categories: "machine-learning"
asset_path: /assets/images/
tags: []
---

# 1. Sentence Bert 

| Key                | Value                            |
|:-------------------|:---------------------------------|
| Paper              | https://arxiv.org/pdf/1908.10084 |
| publication Date   | 2019                             |


## 1.1 Problem 

상황
 - 문장 A 가 있을때 B1 ~ B10,000 까지 1만개의 문장이 있음
 - 이때 A와 가장 유사한 단어를 찾고 싶은 것임

기존 BERT 방식
 - BERT는 문장을 "[CLS] A문장 [SEP] B문장" 형태로 합쳐서 넣고
 - 두 쌍을 BERT에 통채로 넣어서 inference를 함
 - 즉 training 시에 모든 문장의 유사도 계산시 10,000 x 9,999 번의 계산이 필요함
 - 따라서 매우 비효율적임



## 1.2 BERT VS Sentence-BERT (SBERT)

| Key                  | BERT                                       | Setence-BERT (SBERT)                            |
|:---------------------|:-------------------------------------------|:------------------------------------------------|
| input method         | Cross Encoder 방식 - "[CLS] A 문장 [SEP] B 문장" | 각 문장을 따로따로 입력 (Bi-Encoder)                      |
| Embedding Generation | 토큰 단위 (shape: [1, seq_len, 768])           | 문장 전체를 하나의 벡터로 유지 (shape: [384,] <- 벡터 하나)      |
| Performance          | Cross Encoder 방식이 더 정확함. 하지만 느림            | 정확도가 꽤나 잘 유지, 매우 빠름, 특히 검색에서 좋음                 |
| Architecture         | 단일 Transformer                             | Siamese 또는 Triplet Network                      |
| Usage                | Classification, QnA, NLI                   | Similarity Search, Clustering, Recommendation   |



## 1.3 Model 

<img src="{{ page.asset_path }}sbert-model.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

특징
 - BERT 또는 RoBERTa 에 적용 가능
 - Pooling 사용하여 fixed sized sentence embedding 생성
 - Pooling Method
   - CLS Pooling: last_hidden_state[:, 0, :] # shape (1, hidden_dim)
   - Mean Pooling: [PAD] Token을 제외하고 평균을 냄. SBERT 의 기본값
   - MAX Pooling: [PAD] 부분은 음수로 바꾸고 나머지에서 MAX를 취함
 - Training Method: Siamese, Triplet Loss
 - Performance: Bi-Encoder 를 사용함으로서 빠른 유사도 계산 가능


## 1.4 Siamese Network

**Siamese Network (샴 네트워크)**는 두개 이상의 입력 (두 문장, 두 이미지)를 **동일한 모델**에 넣어 각각의 embedding을 만든후,
두 embeddings 의 유사도를 계산하는 구조

수학적 정리
 - 동일한 신경만 f(x) 를 두번 사용
 - 유사도 계산
   - Cosine Similarity: $$ cosine \left( f(x_1), f(x_2) \right) = \frac{ f(x_1) \dot f(x_2)}{ || f(x_1} || || f(x_2) || $$
   - Euclid Distance: $$ || f(x_1) - f(x_2) ||_2 $$

