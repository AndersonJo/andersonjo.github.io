---
layout: post 
title:  "Byte Pair Encoding"
date:   2022-01-15 01:00:00 
categories: "nlp"
asset_path: /assets/images/ 
tags: []
---

# 1. Introduction 

Subword Segmentation은 OOV (out-of-vocabulary problem)문제를 해결하기 위해서 나온 방식입니다. <br> 
즉 기존의 ML의 방식은 단어 하나 하나를 최소 단위로 두고 학습을 했는데, 단어 집합에 없는 케이스도 있으며, <br>
새로운 단어가 나왔을시 대응이 어려워지는 단점이 있었습니다. 

단어 기반의 문제를 해결하기 위해 나온것이 subword segmentation 이고,<br>
기본적으로 단어보다 더 작은 단위로 잘라서 하나의 단어처럼 사용하는 것 입니다. 

대표적인 알고리즘이 구글의 sentence piece, opennmt, huggingface의 tokenizers 등이 있습니다.<br>
본문에서는 그 중 가장 기초적인 알고리즘인 BPE (Byte Pair Encoding)을 다룹니다. 



# 2. Byte Pair Encoding (BPE)

Byte pair encoding은 1994년에 제안된 데이터 압축 알고리즘이며, subword 알고리즘으로 응용이 되었습니다. <br>
알고리즘은 가장 많이 사용되는 pair of bytes를 사용하지 않는 하나의 단어로 바꾸는 것이며, 이것을 n번에 걸쳐서 반복적으로 줄이게 됩니다. <br>
예를 들어서 ...

- `aaabdaaabac` 의 경우 `aa` 가 가장 많이 나오며, 사용하지 않는 단어 Z 로 변환합니다. 
- 변환된 단어는 `ZabdZabac` 이며, 다시 가장 빈번도가 높은 pair of bytes는 `ab` 이며 `Y`로 변환한다면 `ZYdZYac`로 압축될수 있습니다. 



## 2.1 Build Dictionary 

먼저 dictionary 를 만들어야 하며, 이는 기존의 word 단위의 dictionary 와 동일합니다.

- 데이터: corpus는 네이버에서 스파이더맨 (노 웨이 홈)의 평론가의 코멘트, 그리고 나무위키의 내용을 가져왔습니다.
- preprocess 함수: konlpy를 통해서 필요없는 품사들을 제거
- build_dictionary 함수: word 단위의 dictionary 를 구현

{% highlight python %}
import pandas as pd
import numpy as np
import re

from typing import List, Dict
from collections import defaultdict
from konlpy.tag import Okt

okt = Okt()

def preprocess(text) -> List[str]:
    morphs = okt.pos(text)
    tokens = []
    for word, pos in morphs:
        if pos in ('Josa', 'Punctuation', 'Foreign'):
            continue
        
        tokens.append(word)
    return tokens
    

def build_dictionary(corpus: List[str], end_token='\0') -> Dict[str, int]:
    dictionary = defaultdict(int)
    for line in corpus:
        tokens = preprocess(line)
        for token in tokens:
            if not token:
                continue
            dictionary[tuple(list(token) + [end_token])] += 1
    return dictionary


corpus = [
    "스파이더맨, 슈퍼맨, 배트맨, 아이언맨은 올해 최고의 영화가 아닐까 생각함",
    "스파이더맨 파프롬홈은 MCU 최고의 영화이다",
    "슈퍼맨 vs 배트맨의 스토리는 진부하기 짝이 없음. 그리고 어제 IMAX 스파이더맨봤는데 쩔어!",
    "어제 스파이더맨을 IMAX 영화관에서 봤는데.. ㅋㅋ 개쩔어.ㅋㅋㅋㅋ 아이언맨 또 영화관에서 보고 싶다",
    "스파이더짱, 슈가짱, 슈퍼마켓, 배트걸, 아이언돔 ㅋㅋㅋ",
    "스파, 슈가, 슈퍼, 배트, 아이언 ㄹㅇ쩔어! ㅋ",
    "어제 배트맨 봤는데 ㅠㅠ ㄹㅇ 돈날림ㅠㅠ!!"
]

dictionary = build_dictionary(corpus)
{% endhighlight %}


{% highlight python %}
{('스', '파', '이', '더', '맨', '\x00'): 4,
 ('슈', '퍼', '맨', '\x00'): 2,
 ('배', '트', '맨', '\x00'): 3,
 ('아', '이', '언', '맨', '\x00'): 2,
 ('올', '해', '\x00'): 1,
 ('최', '고', '\x00'): 2,
 ('영', '화', '\x00'): 2,
 ('아', '닐', '까', '\x00'): 1,
 ('생', '각', '\x00'): 1,
 ('함', '\x00'): 1,
 ('파', '\x00'): 1,
 ('프', '롬', '\x00'): 1,
 ('홈', '\x00'): 1,
 ('M', 'C', 'U', '\x00'): 1,
 ('v', 's', '\x00'): 1,
 ('스', '토', '리', '\x00'): 1,
 ('진', '부', '하', '기', '\x00'): 1,
 ('짝', '\x00'): 1,
 ('없', '음', '\x00'): 1,
 ('그', '리', '고', '\x00'): 1,
 ('어', '제', '\x00'): 3,
 ('I', 'M', 'A', 'X', '\x00'): 2,
 ('봤', '는', '데', '\x00'): 3,
 ('쩔', '어', '\x00'): 3,
 ('영', '화', '관', '\x00'): 2,
 ('ㅋ', 'ㅋ', '\x00'): 1,
 ('개', '\x00'): 1,
 ('ㅋ', 'ㅋ', 'ㅋ', 'ㅋ', '\x00'): 1,
 ('또', '\x00'): 1,
 ('보', '고', '\x00'): 1,
 ('싶', '다', '\x00'): 1,
 ('스', '파', '이', '더', '\x00'): 1,
 ('짱', '\x00'): 2,
 ('슈', '가', '\x00'): 2,
 ('슈', '퍼', '마', '켓', '\x00'): 1,
 ('배', '트', '\x00'): 2,
 ('걸', '\x00'): 1,
 ('아', '이', '언', '돔', '\x00'): 1,
 ('ㅋ', 'ㅋ', 'ㅋ', '\x00'): 1,
 ('스', '파', '\x00'): 1,
 ('슈', '퍼', '\x00'): 1,
 ('아', '이', '언', '\x00'): 1,
 ('ㄹ', 'ㅇ', '\x00'): 2,
 ('ㅋ', '\x00'): 1,
 ('ㅠ', 'ㅠ', '\x00'): 2,
 ('돈', '\x00'): 1,
 ('날', '림', '\x00'): 1}
{% endhighlight %}



## 2.2 Byte Pair Encoding 

알고리즘의 순서는 다음과 같습니다. 

1. 모든 pair of bytes에 대해서 count 계산을 합니다. 
2. 가장 빈도수가 높은 pair of byte 를 찾습니다.
3. 해당 빈도수 높은 pair of byte를 기존 vocabulary keys 에서 포함되어 있는 단어를 찾습니다. 
    - 만약 포함되어 있는 단어를 찾으면 해당 빈도수 높은 단어로 대체를 하고 기존 단어는 vocabulary 에서 제외 시킵니다. 
    - 만약 찾지 못한다면 기존 단어를 그대로 vocabulary 에 넣습니다. 
4. 다시 1번으로 돌아거 n번만큼 반복합니다. 






{% highlight python %}
def get_pairs(dictionary, end_token='\0'):
    pairs = defaultdict(int)
    for word, freq in dictionary.items():
        i = 0  # len(word) == 1 인 경우 i의 값이 초기화가 안되기 때문에 필요
        for i in range(len(word)-1):
            pairs[word[i], word[i+1]] += freq
    return pairs

def search_merge_vocab(pair, dictionary, end_token='\0') -> Dict[str, int]:
    vocab = {}
    pair_word = ''.join(pair)
    
    for word in dictionary:    
        i = 0
        flag = False
        key = list()
        
        while i < (len(word)):
            if (pair[0] == word[i]) and pair[1] == word[i+1]:
                key.append(pair_word)
                i += 2
                flag = True
            else:
                key.append(word[i])
                i += 1
                
        vocab[tuple(key)] = dictionary[word]
    return vocab

def bpe(dictionary, n_iter=10):
    for i in range(n_iter):
        pairs = get_pairs(dictionary)
        most_freq_pair = max(pairs, key=pairs.get)
        dictionary = search_merge_vocab(most_freq_pair, dictionary)
    return dictionary
bpe(dictionary, n_iter=50)
{% endhighlight %}

{% highlight python %}
{('스파이더맨\x00',): 4,
 ('슈퍼맨\x00',): 2,
 ('배트맨\x00',): 3,
 ('아이언맨\x00',): 2,
 ('올해\x00',): 1,
 ('최고\x00',): 2,
 ('영화\x00',): 2,
 ('아닐까\x00',): 1,
 ('생각\x00',): 1,
 ('함\x00',): 1,
 ('파\x00',): 1,
 ('프롬', '\x00'): 1,
 ('홈', '\x00'): 1,
 ('M', 'C', 'U', '\x00'): 1,
 ('v', 's', '\x00'): 1,
 ('스', '토', '리', '\x00'): 1,
 ('진', '부', '하', '기', '\x00'): 1,
 ('짝', '\x00'): 1,
 ('없', '음', '\x00'): 1,
 ('그', '리', '고\x00'): 1,
 ('어제\x00',): 3,
 ('IMAX\x00',): 2,
 ('봤는데\x00',): 3,
 ('쩔어\x00',): 3,
 ('영화관\x00',): 2,
 ('ㅋㅋ\x00',): 1,
 ('개', '\x00'): 1,
 ('ㅋㅋ', 'ㅋㅋ\x00'): 1,
 ('또', '\x00'): 1,
 ('보', '고\x00'): 1,
 ('싶', '다', '\x00'): 1,
 ('스파이더', '\x00'): 1,
 ('짱\x00',): 2,
 ('슈가\x00',): 2,
 ('슈퍼', '마', '켓', '\x00'): 1,
 ('배트\x00',): 2,
 ('걸', '\x00'): 1,
 ('아이언', '돔', '\x00'): 1,
 ('ㅋㅋ', 'ㅋ\x00'): 1,
 ('스파', '\x00'): 1,
 ('슈퍼', '\x00'): 1,
 ('아이언', '\x00'): 1,
 ('ㄹㅇ\x00',): 2,
 ('ㅋ\x00',): 1,
 ('ㅠㅠ\x00',): 2,
 ('돈', '\x00'): 1,
 ('날', '림', '\x00'): 1}
{% endhighlight %}