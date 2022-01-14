---
layout: post 
title:  "KServe Custom Model"
date:   2022-01-01 01:00:00 
categories: "kubernetes"
asset_path: /assets/images/ 
tags: ['pack']
---


# 1. Installation

## 1.1 Pack 

Pack CLI 설치가 먼저 필요합니다. <br>
자세한 설치문서는 [링크](https://buildpacks.io/docs/tools/pack/) 를 참조합니다.


{% highlight bash %}
sudo add-apt-repository ppa:cncf-buildpacks/pack-cli
sudo apt-get update
sudo apt-get install pack-cli
{% endhighlight %}


`.bashrc` 에 다음을 추가합니다. <br>
Autocomplete 기능이 추가가 됩니다. 

{% highlight bash %}
# Pack Autocompletion
. $(pack completion)
{% endhighlight %}


## 1.2 Install KServe 

{% highlight bash %}
pip install --upgrade kserve google-api-core 
{% endhighlight %}



# 2. Custom Model

Kserve.Model 이 base class 이고, `preprocess`, `predict`, 그리고 `postprocess` 핸들러를 정의하고 있으며, 순서대로 실행됩니다. <br>


{% highlight python %}
cat <<EOF > app.py
import kserve
from typing import Dict

class AlexNetModel(kserve.Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.load()

    def load(self):
        pass

    def predict(self, request: Dict) -> Dict:
        pass

if __name__ == "__main__":
    model = AlexNetModel("custom-model")
    kserve.ModelServer().start([model])
EOF
{% endhighlight %}