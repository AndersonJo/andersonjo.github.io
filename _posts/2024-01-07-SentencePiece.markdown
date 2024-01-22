---
layout: post
title:  "SentencePiece"
date:   2024-01-07 01:00:00
categories: "nlp"
asset_path: /assets/images/
tags: []
---

# Sentence Piece 

## Installation

```bash
$ pip install --upgrade sentencepiece datasets transformers

```

## Data


```python
dataset = load_dataset("nsmc")

data_path = Path('/tmp/sentencepiece-data')
os.makedirs(data_path, exist_ok=True)

for key in dataset.keys():
    file_path = data_path / f'{key}.txt'
    print(file_path)
    with open(file_path, 'wt') as f:
        for line in dataset[key]['document']:
            f.write(f'{line}\n')
```

아래와 같이 파일이 생성됩니다. 

```text
아 더빙.. 진짜 짜증나네요 목소리
흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나
너무재밓었다그래서보는것을추천한다
교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정
사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다
막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.
원작의 긴장감을 제대로 살려내지못했다.
```


## Train

| Option               | Description                                                                                                                                                      |
|:---------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --input              | , (comma) 로 구분되는 파일 위치를 넣으면 됩니다. (ex. "A.txt,B.txt,C.txt"). <br>또한 SentencePiece는 NFKC 로 normalize 하기 때문에 따로 tokenizer, normalizer 또는 preprocessor 를 돌릴필요가 없습니다. |
| --model_prefix       | model name prefix                                                                                                                                                |
| --vocab_size         | e.g. 8000, 16000 or 32000                                                                                                                                        |
| --character_coverage | default is 0.9995                                                                                                                                                |
| --model_type         | unigram (default), bpe, char, word (word 사용시 pretokenized 되 있어야 합니다.)                                                                                            |
| --num_threads        | unigram 에서만 작동합니다.                                                                                                                                               |

> Jupyter Notebook 에서 실행시키면, logging이 보이지 않습니다. <br> 
> 따라서 python 파일 따로 만들어서 실행시키는 것을 추천합니다. (notebook 에서는 심지어 멈추기 까지 합니다.)

학습시간은 대충 1분도 안걸립니다.

```python
import sentencepiece as spm

corpus = ",".join([str(x) for x in data_path.glob("*.txt")])
vocab_size = 31900
model_prefix = "sp-bpt-nsmc"

spm.SentencePieceTrainer.train(
    input=corpus,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type="bpe",
    max_sentence_length=500000,
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    user_defined_symbols="<sep>,<cls>,<mask>",
    num_threads=16
)
```

또는 다음과 같이 실행합니다. 

```python
spm.SentencePieceTrainer.train(
    f"--input={corpus} "
    f"--model_prefix={model_prefix} "
    f"--vocab_size={vocab_size} "
    f"--model_type=bpe "
    f"--max_sentence_length=500000 "
    " --character_coverage=1.0 "
    " --pad_id=0 --pad_piece=<pad> "
    " --unk_id=1 --unk_piece=<unk> "  # Unknown
    " --bos_id=2 --bos_piece=<s> "  # begin of sequence
    " --eos_id=3 --eos_piece=</s> "  # end of sequence
    " --user_defined_symbols=<sep>,<cls>,<mask> "
    " --num_threads=16 "
)
```


## Encoding & Decoding 

학습이 모두 끝났으면 불러와서 사용하면 됩니다. <br>
아래처럼 모델을 불러오면 됩니다.

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('sp-bpt-nsmc.model')
```

```python
example_sentence = '홍콩반점 짬뽕은 맛있다 하지만 오끽궁보다 못하다'

# Encoding
sp.Encode(example_sentence)
sp.EncodeAsIds(example_sentence)
# [1786, 28852, 28749, 7137, 28719, 1278, 646, 515, 77, 30267, 29258, 114, 4830]

# Tokenizing
sp.EncodeAsPieces(example_sentence)
# ['▁홍콩', '반', '점', '▁짬뽕', '은', '▁맛', '있다', '▁하지만', '▁오', '끽', '꾝', '보다는', '▁못하다']


# Decoding
encoded = [1786, 28852, 28749, 7137, 28719, 1278, 646, 515, 77, 30267, 29258, 114, 4830]
sp.Decode(encoded)
sp.DecodeIds(encoded)
# '홍콩반점 짬뽕은 맛있다 하지만 오끽궁보다 못하다'
```

## Token Information

```python
# Vocab Size
vocab_size = sp.get_piece_size()  # 31900

# List of Tokens and ID
for i in range(sp.get_piece_size()):
    print(f'{i:05}: {sp.id_to_piece(i)}')

    if i > 100:
        break

# 00000: <pad>
# 00001: <unk>
# 00002: <s>
# 00003: </s>
# 00004: <sep>
# 00005: <cls>
# 00006: <mask>
# 00007: ..
# 00008: 영화
# 00009: ▁영화
# 00010: ▁이
# 00011: ▁아
```