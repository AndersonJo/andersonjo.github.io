---
layout: post 
title:  "Sentence Piece Tokenizer"
date:   2022-01-18 01:00:00 
categories: "nlp"
asset_path: /assets/images/ 
tags: ['bpe']
---


# 1. Sentece Piece 

- unsupervised text tokenizer and detokenizer 이며 주로 딥러닝에서 사용
- 내부적으로 BPE (byte-pair-encoding) 그리고 unigram language model 을 사용
- 특정 언어에 국한되지 않고, 다양한 언어에 사용 가능

- [논문](https://arxiv.org/pdf/1808.06226.pdf)
- [Github](https://github.com/google/sentencepiece)

설치

{% highlight bash %}
$ sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
$ pip install sentencepiece
{% endhighlight %}


초기화

{% highlight python %}
from pathlib import Path
from tempfile import gettempdir
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import sentencepiece as stp
from konlpy.tag import Okt

okt = Okt()
{% endhighlight %}



# 2. Sentencepiece in Python

## 2.1 Train with text File 

- input: 학습 파일 위치
- model_prefix: 모델이름
- vocab_size: vocabulary 단어 크기
- model_type: `unigram` (default) | `bpe` | `char` | `word`
- max_sentence_length: 문장 최대 길이
- pad_id: pad token ID
- unk_id: unknown token ID
- bos_id: Begin of sentence token ID
- eos_id: End of sentence token ID 
- user_defined_symbols: 사용자 정의 토큰


{% highlight python %}
train_morph_path = Path(gettempdir()) / "sentencepiece-train.txt"
model_prefix_path = Path(gettempdir()) / "nsmc-sentencepiece"
train_df.morph.to_csv(train_morph_path, index=False, header=False)

stp.SentencePieceTrainer.train(
    input=train_morph_path,
    model_prefix=model_prefix_path,
    vocab_size=4000,
    user_defined_symbols=["foo", "bar"],
)
{% endhighlight %}


## 2.2 Encoding and Decoding

**텍스트 한개의 경우**

 - sentencepiece.Encode(text) 는 EncodeAsIds 와 동일
 - sentencepiece.EncodeAsIds(text) -> [12, 14, 2, 3, ...]
 - sentencepiece.EncodeAsPieces(text) -> ['▁잼있', '고', '▁신나', '는', ...]
 - sentencepiece.Encode(encoded_text) -> "잼있고 신나는 ..."

{% highlight python %}
sp = stp.SentencePieceProcessor()
sp.load(str(model_prefix_path.with_suffix(".model")))

text = test_df.sample().morph.values[0]
print('Text             :', text)
print('Encode as IDs   :', sp.EncodeAsIds(text))
print('Encode as Pieces:', sp.EncodeAsPieces(text))
print('Decode from IDs :', sp.decode(sp.Encode(text)))
{% endhighlight %}

{% highlight bash %}
Text             : 잼있고 신나는 영화 보드 타는 모습 너무 멋져
Encode as IDs   : [668, 15, 3089, 19, 7, 118, 145, 501, 19, 421, 23, 1139, 394]
Encode as Pieces: ['▁잼있', '고', '▁신나', '는', '▁영화', '▁보', '드', '▁타', '는', '▁모습', '▁너무', '▁멋', '져']
Decode from IDs : 잼있고 신나는 영화 보드 타는 모습 너무 멋져
{% endhighlight %}


**텍스트가 여러개인 경우**

{% highlight python %}
text_list = test_df.sample(3).morph.tolist()

print('[Text]')
display(text_list)

print('\n[Encode]')
encoded = sp.encode(text_list)
print(encoded)

print('\n[Encode as Pieces]')
print([sp.encode_as_pieces(line) for line in text_list])

print('\n[Decode]')
sp.decode(encoded)
{% endhighlight %}

{% highlight bash %}
[Text]
['주인공 들 너 무답 답 미 리터 놓고말 좀하지 특히 여 주인공',
 '진짜 재밌게 봤고 다시 봐두 재미 에 감동 임창정 짱 ㅠㅂㅠ',
 '그때 나 지금 이나 군대 는']

[Encode]
[[195, 13, 680, 116, 1457, 983, 187, 480, 287, 5, 3993, 15, 413, 111, 889, 607, 146, 195], [34, 236, 5, 3999, 15, 143, 497, 806, 102, 10, 80, 171, 980, 160, 279, 328, 3196], [1309, 30, 205, 323, 688, 158, 14]]

[Encode as Pieces]
[['▁주인공', '▁들', '▁너', '▁무', '답', '▁답', '▁미', '▁리', '터', '▁', '놓', '고', '말', '▁좀', '하지', '▁특히', '▁여', '▁주인공'], ['▁진짜', '▁재밌게', '▁', '봤', '고', '▁다시', '▁봐', '두', '▁재미', '▁에', '▁감동', '▁임', '창', '정', '▁짱', '▁ᅲ', '뷰'], ['▁그때', '▁나', '▁지금', '▁이나', '▁군', '대', '▁는']]

[Decode]
['주인공 들 너 무답 답 미 리터 놓고말 좀하지 특히 여 주인공',
 '진짜 재밌게 봤고 다시 봐두 재미 에 감동 임창정 짱 ᅲ뷰',
 '그때 나 지금 이나 군대 는']
{% endhighlight %}