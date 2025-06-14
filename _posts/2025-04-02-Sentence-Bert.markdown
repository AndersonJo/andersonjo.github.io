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

| Key                  | BERT                                                            | Setence-BERT (SBERT)                            |
|:---------------------|:----------------------------------------------------------------|:------------------------------------------------|
| input method         | 두 문장을 한 문장에 넣어서 입력 (Cross Encoder 방식) - "[CLS] A 문장 [SEP] B 문장" | 각 문장을 따로따로 입력 (Bi-Encoder)                      |
| Embedding Generation | 토큰 단위 -> 즉 전체 문장 X (shape: [1, seq_len, 768])                   | 문장 전체를 하나의 벡터로 유지 (shape: [384,] <- 벡터 하나)      |
| Performance          | Cross Encoder 방식이 더 정확함. 하지만 느리고, 트레이닝도 힘듬                      | 정확도가 꽤나 잘 유지, 매우 빠름, 특히 검색에서 좋음                 |
| Architecture         | 단일 Transforemr                                                  | Siamese 또는 Triplet Network                      |
| Usage                | Classification, QnA, NLI                                        | Similarity Search, Clustering, Recommendation   |


## 