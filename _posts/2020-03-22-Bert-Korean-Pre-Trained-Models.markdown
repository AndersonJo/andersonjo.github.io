---
layout: post
title:  "Korean Pre-Trained Models & Datasets"
date:   2020-03-22 01:00:00
categories: "nlp"
asset_path: /assets/images/
tags: ['kobert']
---

# 1. Bert

| Model                                                                | Vocab | Layers        | NSMC ACC | Naver-NER F1 | Corpus                                                                                      |
|:---------------------------------------------------------------------|:------|:--------------|:---------|:-------------|:--------------------------------------------------------------------------------------------|
| [KoBERT](https://github.com/SKTBrain/KoBERT)                         | 8002  | 12 layers     | 89.63    | 86.11        | Wiki \n - 5M sentences \n - 54 words                                                        |
| [KoSentenceBert - ETRI](https://github.com/BM-K/KoSentenceBERT-ETRI) |       |               |          |              |                                                                                             |
| [LMKor](https://github.com/kiyoungkim1/LMkor)                        | 42000 | 6, 12 layers  | 90.87    | 87.27        | - 국내 주요 커머스 리뷰 1억개 + 블로그 형 웹사이트 2000만개 (75GB) \n - 모두의 말뭉치 (18GB) \n - - 위키피디아와 나무위키 (6GB)  |


# 2. Datasets

| Name                                                                   | Type                 | Description                                            | 
|:-----------------------------------------------------------------------|:---------------------|:-------------------------------------------------------|
| [SNLI Dataset Corpus](https://nlp.stanford.edu/projects/snli/)         | Sentence Similarity  | 2개의 영어 문장이 있고, contradiction, neutral, entailment 로 나눔 |
| [The Multi-Genre NLI Corpus](https://cims.nyu.edu/~sbowman/multinli/)  | Sentence Similarity  | 2개의 영어 문장이 있고, contradiction, neutral, entailment 로 나눔 |
