---
layout: post
title:  "Ray Serving on Kubernetes"
date:   2022-08-15 01:00:00
categories: "ml-ops"
asset_path: /assets/images/
tags: ['ray', 'kubernetes']
---

# 1. Ray Serve

## 1.1 Installation

```bash
$ pip install transformers requests huggingface_hub sentencepiece
$ pip install ray[serve] 
```

## 1.2 Serving Example

`app.py` 를 만들고 다음과 같이 코드를 작성합니다.

```python
from ray import serve
from starlette.requests import Request
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.5, "num_gpus": 0})
class Translator:
    def __init__(self):
        # Load model
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
        self.tokenizer.src_lang = 'en'

        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
        self.model.eval()

    def translate(self, text: str) -> str:
        dest_lang_id = self.tokenizer.get_lang_id('ko')
        encoded_src = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_src,
                                               forced_bos_token_id=dest_lang_id,
                                               max_length=200,
                                               use_cache=True)
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return result

    async def __call__(self, http_request: Request) -> str:
        korean_text: str = await http_request.json()
        return self.translate(korean_text)


translator = Translator.bind()

# if __name__ == '__main__':
#     translator = Translator()
#     print(translator.translate('self-belief and hard work will always earn you success'))
```


Ray Serving 은 다음과 같이 합니다.

```bash
# Server 올리기
$ serve run app:translator

# 테스트
$ curl localhost:8000 -H "Accept: application/json" \
    -d '"self-belief and hard work will always earn you success"'
    
자신감과 열심히 일하면 항상 당신에게 성공을 가져올 것입니다.
```

