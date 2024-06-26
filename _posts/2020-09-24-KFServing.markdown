---
layout: post
title:  "KFServing on EKS"
date:   2020-09-24 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['aws', 'machine-learning', 'ml-ops', 'seldon-core', 'mlops', 'autoscaling']
---

# 1. Setting Up!

## 1.1 Before Installation

해당 문서는 2022년 01월 기준으로 썼으며 (계속 업데이트중), KServing 0.7 버젼을 기준으로 합니다.  

 1. Kubeflow 설치 필요 없음 (하지마)
 2. Cert Manager (v1.0 이상) 설치 필요함 
 3. Istio (latest) 설치해야 함 
 4. Knative Serving (latest) 설치 해야 함
    - Knative Eventing 설치 필요 없음 (하지마!)

추가적으로 현재 문서에서 KFServing 으로 되어 있는데, 0.7버젼으로 가면서 KServe 로 이름 변경함<br>
나는 안 바꿨음.. 

## 1.2 Kuberflow 와 비교

일단 kubeflow는 사용하지 마세요.<br>
그 안에 뭐 pipeline이나 notebook 기능 등등 다채로운 기능들 하나로 다 짬뽕 시켜놓은 것인데.. <br>
문제는 그렇게 짬뽕 시켜놨으면 관리가 잘 되야 하는데 안되고 있어요.<br>
버젼 업그레이드도 느리고 .. 그냥 KFServing 만 사용하는 것을 추천 합니다. 


## 1.3 Install Serverless KFServing  

먼저 Python SDK를 설치합니다.

{% highlight bash %}
$ pip install kfserving
{% endhighlight %}

[Serverless Mode](https://kserve.github.io/website/admin/serverless/) 를 설치합니다. <br>
Serverless Mode는 반드시 Knative가 base를 이루고 있으며, Knative의 제약을 받습니다.<br>
제약을 피하고자 한다면 [Kubernetes Deployment Installation](https://kserve.github.io/website/admin/kubernetes_deployment/)을 참조 합니다. 

{% highlight bash %}
# Serverless Model Installation
$ kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.7.0/kserve.yaml
{% endhighlight %}




## 1.4 ~~Serving namespace 지정~~

Kubeflow에서는 이미 KFServing 이 설치되어서 나옵니다.

{% highlight bash %}
$ kubectl create namespace kfserving
$ kubectl label namespace kfserving istio-injection=enabled
$ kubectl label namespace kfserving serving.kubeflow.org/inferenceservice=enabled
$ kubectl get ns kfserving -o json | jq .metadata.labels
{
  "istio-injection": "enabled",
  "serving.kubeflow.org/inferenceservice": "enabled"
}
{% endhighlight %}

KFServing controller 가 설치되어 있는지 확인합니다.

{% highlight bash %}
# Kuberflow 설치시
$ kubectl get po -n kubeflow | grep kfserving-controller-manager
kfserving-controller-manager-0        2/2     Running   0          3h55m

# KFServing 단독 설치시
$ kubectl get po -n kfserving-system | grep kfserving-controller-manager
kfserving-controller-manager-0        2/2     Running   0          56s

{% endhighlight %}


# 2. Iris Tutorial  

### 2.1 Deployment


{% highlight yaml %}
cat <<EOF > sklearn.yaml 
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    sklearn:
      storageUri: "gs://kfserving-samples/models/sklearn/iris"
EOF
{% endhighlight %}

모델 배포

{% highlight bash %}
$ kubectl create namespace kserve-test
$ kubectl apply -f sklearn.yaml -n kserve-test
{% endhighlight %}

**데이터 생성**

{% highlight yaml %}
cat <<EOF > iris-input.json
{
  "instances": [
    [5.0,  3.4,  1.5,  0.2],
    [6.7,  3.1,  4.4,  1.4],
    [6.1,  3.0,  4.9,  1.8]
  ]
}
EOF
{% endhighlight %}

**배포**

{% highlight bash %}
# sklearn InferenceService를 배포합니다. 
$ kubectl apply -f sklearn.yaml -n kserve-test

# Service URL 을 확인합니다. (URL 뜨는데 까지 약 20~30초 걸림)
$ kubectl get inferenceservices sklearn-iris -n kserve-test
NAME           URL                                                                READY   DEFAULT TRAFFIC
sklearn-iris   http://sklearn-iris.kfserving.example.com/v1/models/sklearn-iris   True    100            
{% endhighlight %}

AWS EKS는 ingress 를 외부 연결로 쓰지 않고 따로 LoadBalancer를 지정해줘야 합니다. ㅜㅜ 개불편. <br>
(GCP는 ingress 설정하면 알아서 load balancer 잡힘)

{% highlight bash %}
$ kubectl get svc istio-ingressgateway -n istio-system
NAME                   TYPE       CLUSTER-IP       EXTERNAL-IP   PORT(S)        
istio-ingressgateway   NodePort   10.100.200.170   <none>        15020:30234/TCP <생략>
{% endhighlight %}

필요한건 NodePort를 LoadBalancer로 변경해주면 됩니다. 

{% highlight bash %}
$ kubectl edit svc istio-ingressgateway  -n istio-system
{% endhighlight %}

`type: NodePort` 를 `type: LoadBalancer`로 변경해줍니다.<br>
이후 다시 service를 확인해보면 EXTERNAL-IP가 잡혀있게 됩니다. 

{% highlight bash %}
$ kubectl --namespace istio-system get service istio-ingressgateway
NAME                   TYPE           CLUSTER-IP       EXTERNAL-IP                        PORT(S)                    
istio-ingressgateway   LoadBalancer   10.100.200.170   ****.us-east-2.elb.amazonaws.com   80:32051/TCP,443:31101/TCP
{% endhighlight %}




### 2.1.2 Inference from External Source

 - INGRESS_HOST: Load Balancer Hostname (ex. `a0e3184ae3-1490218815.us-east-2.elb.amazonaws.com`)
 - INGRESS_PORT: Load Balancer Port (ex. `80`)
 - SERVICE_HOSTNAME: 모델 서빙되고 있는 주소 (ex. `sklearn-iris.kfserving.example.com`)

**환경변수 설정**

{% highlight bash %}
$ INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
$ INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
$ SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -n kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
{% endhighlight %}

**Inference**

{% highlight bash %}
$ curl -v -H  "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict -d @./iris-input.json
{"predictions": [0, 1, 2]}
{% endhighlight %}

**PostMan**

 - URL: {INGRESS_HOST}:{INGRESS_PORT}/v1/models/sklearn-iris
 - Headers 추가 
     - Host: {SERVICE_HOSTNAME}
 - Body에 JSON형식으로 데이터 추가

<img src="{{ page.asset_path }}kfserving-postman-01.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

<img src="{{ page.asset_path }}kfserving-postman-02.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

### 2.1.3 Inference from Local Cluster Gateway

Cluster 내부에서의 통신은 위에처럼 외부 load balancer를 타서 통신을 할 필요가 없습니다. <br>
즉 내부 통신을 이용하면 빠르게 데이터 교환을 할 수 있습니다.

먼저 내부에서 통신할 URL을 알아냅니다. 

{% highlight bash %}
$ kubectl get inferenceService -n kserve-test sklearn-iris -o jsonpath='{.status.address.url}' 
http://sklearn-iris.kfserving.svc.cluster.local/v1/models/sklearn-iris:predict
{% endhighlight %}

이제 특정 Container로 접속합니다. <br>
아래는 예제 이며, "sklearn-iris-predictor-default-***" 요 부분은 pod 이름입니다.

{% highlight bash %}
$ kubectl exec -it sklearn-iris-predictor-*** -n kserve-test -c kserve-container /bin/bash
$ curl -i http://sklearn-iris.kserve-test.svc.cluster.local/v1/models/sklearn-iris:predict -d @./iris-input.json
{"predictions": [0, 1, 2]}
{% endhighlight %}




### 2.1.4 Performance Test

위에서 배포한 IRIS 모델의 퍼포먼스를 측정합니다. 

{% highlight bash %}
$ kubectl create -f https://raw.githubusercontent.com/kserve/kserve/release-0.7/docs/samples/v1beta1/sklearn/v1/perf.yaml -n kserve-test
{% endhighlight %}


{% highlight bash %}
$ kubectl logs load-testpk9r2-wmknb -n kfserving 
Requests      [total, rate, throughput]         30000, 500.02, 499.95
Duration      [total, attack, wait]             1m0s, 59.998s, 7.861ms
Latencies     [min, mean, 50, 90, 95, 99, max]  3.293ms, 8.509ms, 5.778ms, 15.34ms, 21.869ms, 47.731ms, 155.37ms
Bytes In      [total, mean]                     690000, 23.00
Bytes Out     [total, mean]                     2460000, 82.00
Success       [ratio]                           100.00%
Status Codes  [code:count]                      200:30000  
Error Set:
{% endhighlight %}













## 2.2 InferenceService with Custom Image

**Flask App**

{% highlight bash %}
cat <<EOF > app.py
import os
from flask import Flask

app = Flask(__name__)

@app.route('/v1/models/custom-image:predict')
def hello_world():
    greeting_target = os.environ.get('GREETING_TARGET', 'World')
    return 'Hello {}!\n'.format(greeting_target)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
EOF
{% endhighlight %}


**requirements.txt**

{% highlight bash %}
cat <<EOF > requirements.txt
Flask==1.1.1
gunicorn==20.0.4
EOF
{% endhighlight %}

**Dockerfile**

{% highlight bash %}
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
CMD exec gunicorn --bind :\$PORT --workers 1 --threads 8 app:app
EOF
{% endhighlight %}

**Docker Build**

 - 아래에 andersonjo 라고 되어 있는 부분은 Docker Hub의 ID를 넣어주시면 됩니다. 

{% highlight bash %}
$ docker login
$ docker build -t andersonjo/custom-image .
$ docker run -d --name custom-test -p 8080:8080 -it test
$ docker push andersonjo/custom-image
{% endhighlight %}


**Custom YAML**

{% highlight yaml %}
cat <<EOF > custom.yaml 
apiVersion: serving.kubeflow.org/v1alpha2
kind: InferenceService
metadata:
  labels:
    controller-tools.k8s.io: "1.0"
  name: custom-image
spec:
  default:
    predictor:
      custom:
        container:
          name: custom
          image: andersonjo/custom-image
          env:
            - name: GREETING_TARGET
              value: "Python KFServing Sample"
EOF
{% endhighlight %}

**Deployment**

{% highlight bash %}
$ kubectl apply -f custom.yaml -n kfserving
$ kubectl get inferenceservices -n kfserving
NAME           URL                                                                READY
custom-image   http://custom-image.kfserving.example.com/v1/models/custom-image   True 
{% endhighlight %}

**Inference**

{% highlight bash %}
$ INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
$ INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
$ SERVICE_HOSTNAME=$(kubectl get inferenceservice custom-image -n kfserving -o jsonpath='{.status.url}' | cut -d "/" -f 3)
$ curl -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/custom-image:predict
Hello Python KFServing Sample!
{% endhighlight %}


## 2.3 Autoscale InferenceService 

### 2.3.1 Create Inference Service

autoscale.yaml inference service 를 생성합니다.

{% highlight yaml %}
cat <<EOF > autoscale.yaml
apiVersion: "serving.kubeflow.org/v1alpha2"
kind: "InferenceService"
metadata:
  name: "flowers-sample"
spec:
  default:
    predictor:
      tensorflow:
        storageUri: "gs://kfserving-samples/models/tensorflow/flowers" 
EOF
{% endhighlight %}

{% highlight json %}
cat <<EOF > input.json
{  
    "instances":[  
       {  
          "image_bytes":{  
             "b64":"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"
          },
          "key":"   1"
       }
    ]
}
EOF
{% endhighlight %}

{% highlight bash %}
$ kubectl apply -f autoscale.yaml -n kfserving
$ kubectl get inferenceservices flowers-sample -n kfserving
NAME             URL                                                                    READY   DEFAULT
flowers-sample   http://flowers-sample.kfserving.ai.platform/v1/models/flowers-sample   True    100    
{% endhighlight %}

### 2.3.2 Stress Test 

hey 명령어를 통해서 스트레스 테스트를 가볍게 해볼수 있습니다.

 - **-z**: duration이고 10s는 10초, 3m 은 3분  
 - **-c**: 동시 requests 갯수. concurrent requests 이며 전체 requests갯수는 아님
 - **-m**: HTTP method. POST, GET, PUT, DELETE, HEAD, OPTIONS .. 

{% highlight bash %}
$ MODEL_NAME=flowers-sample
$ INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
$ INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
$ SERVICE_HOSTNAME=$(kubectl get inferenceservice flowers-sample -n kfserving -o jsonpath='{.status.url}' | cut -d "/" -f 3)
$ hey -z 30s -c 5 -m POST -host ${HOST} -D $INPUT_PATH http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/$MODEL_NAME:predict
{% endhighlight %}

결과

{% highlight bash %}
Summary:
  Total:	30.1505 secs
  Slowest:	0.4009 secs
  Fastest:	0.1899 secs
  Average:	0.1944 secs
  Requests/sec:	25.6712
  
  Total data:	133128 bytes
  Size/request:	172 bytes

Response time histogram:
  0.190 [1]	|
  0.211 [768]	|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.232 [0]	|
  0.253 [0]	|
  0.274 [0]	|
  0.295 [0]	|
  0.316 [0]	|
  0.338 [0]	|
  0.359 [0]	|
  0.380 [0]	|
  0.401 [5]	|


Latency distribution:
  10% in 0.1911 secs
  25% in 0.1916 secs
  50% in 0.1927 secs
  75% in 0.1944 secs
  90% in 0.1955 secs
  95% in 0.1963 secs
  99% in 0.2001 secs

Details (average, fastest, slowest):
  DNS+dialup:	0.0013 secs, 0.1899 secs, 0.4009 secs
  DNS-lookup:	0.0000 secs, 0.0000 secs, 0.0070 secs
  req write:	0.0000 secs, 0.0000 secs, 0.0002 secs
  resp wait:	0.1929 secs, 0.1897 secs, 0.2022 secs
  resp read:	0.0001 secs, 0.0000 secs, 0.0002 secs

Status code distribution:
  [404]	774 responses
{% endhighlight %}
