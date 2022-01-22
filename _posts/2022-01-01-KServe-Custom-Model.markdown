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

## 2.1 Custom Inference Code 

Kserve.Model 이 base class 이고, `preprocess`, `predict`, 그리고 `postprocess` 핸들러를 정의하고 있으며, 순서대로 실행됩니다. <br>
`predict` 함수에서 inference를 실행시켜야 하며, `postprocess` 에서 raw prediction 결과를 user-friendly response 로 변형을 해 줍니다.<br>
기본적인 서버 정보는 다음과 같습니다.

 - default port: 8080
 - default grpc port: 8081

app.py

{% highlight python %}
cat <<EOF > app.py
import kserve
from typing import Dict

class CustomModel(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name

    def preprocess(self, request: Dict) -> Dict:
        print('preprocess', request)
        data = request['data']
        return data

    def predict(self, request: Dict) -> Dict:
        print('predict', request)
        return sum(request)

    def postprocess(self, request: Dict) -> Dict:
        print('postprocess', request)
        return {'result': request}

if __name__ == "__main__":
    model = CustomModel("api/anderson-custom-model")
    kserve.KFServer(workers=1).start([model], nest_asyncio=True)
EOF
{% endhighlight %}

requirements.txt

{% highlight python %}
cat <<EOF > requirements.txt
kserve==0.7.0
nest-asyncio==1.5.4
EOF
{% endhighlight %}

input.json

{% highlight python %}
cat <<EOF > input.json
{
  "data": [13, 20, 45, 5]
}
EOF
{% endhighlight %}

테스트는 먼저 서버를 띄워놓고 request를 날려봅니다. <br>
내부적으로 tornado server 가 띄워집니다. 

{% highlight python %}
$ python app.py
$ curl localhost:8080/v1/models/anderson-custom-model:predict -d @./input.json
{"result": 83}
{% endhighlight %}




## 2.2 Docker Build

{% highlight yaml %}
cat <<EOF > Dockerfile
FROM python:3.7-slim

ENV APP_HOME=/app
WORKDIR \$APP_HOME
COPY app.py requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD ["python", "app.py"]
EOF
{% endhighlight %}


{% highlight bash %}
# 빌드
$ docker build -t andersonjo/kserve-custom-model:v1 .

# 테스트
$ docker run -p 8080:8080 -t andersonjo/kserve-custom-model:v1
$ curl localhost:8080/v1/models/anderson-custom-model:predict -d @./input.json
{% endhighlight %}


## ~~2.2 Build the custom image with Buildpacks~~

[Buildpacks](https://buildpacks.io/)은 위의 inference code를 Dockerfile 작성없이 배포가능한 docker image 로 변형시켜줍니다.<br>
Buildpack은 자동으로 Python application을 확인한뒤, requirements.txt에 있는 dependencies를 설치해줍니다. <br>

아래 andersonjo 부분은 docker hub의 user name 으로 변경하면 됩니다.

{% highlight bash %}
# 먼저 builder 추천을 받습니다. 
$ pack builder suggest
$ pack build --builder=heroku/buildpacks:20 andersonjo/kserve-custom-model:v1
$ docker push andersonjo/kserve-custom-model:v1
{% endhighlight %}


## InferenceService (Deployment)



{% highlight python %}
cat <<EOF > custom-model.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: custom-model
  namespace: kserve-test
spec:
  predictor:
    containers:
      - name: kserve-container
        image: andersonjo/kserve-custom-model:v1
EOF
{% endhighlight %}


{% highlight python %}
$ kubectl apply -f custom-model.yaml
{% endhighlight %}