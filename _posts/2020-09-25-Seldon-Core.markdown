---
layout: post
title:  "Seldon Core"
date:   2020-09-25 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['kubernetes']
---


# 1. Install Seldon Core with Kubeflow 

## 1.1 Uninstall Existing Seldon Core 

먼저 namespace 삭제해줍니다.

{% highlight bash %}
$ kubectl get ns seldon-system -o json | jq '.spec.finalizers'=null | kubectl apply -f -
$ kubectl delete ns/seldon-system
{% endhighlight %}

Custom Resources Definitions 삭제는 다음과 같이 합니다. 

{% highlight bash %}
$ kubectl get crd | grep seldon
NAME                                                 CREATED AT
.. 생략
seldondeployments.machinelearning.seldon.io          2020-10-05T13:49:42Z

$ kubectl delete crd/seldondeployments.machinelearning.seldon.io
{% endhighlight %}

## 1.2 Install Seldon Core with Helm 

먼저 dependencies를 설치합니다.  

{% highlight bash %}
# Ubuntu 
$ sudo snap install helm --classic
$ sudo snap install kustomize

$ helm version --short  # 3.x 버젼을 확인합니다.
v3.3.4+ga61ce56
{% endhighlight %} 

이후에 Seldon Core를 helm으로 설치 합니다.

{% highlight bash %}
$ kubectl create namespace seldon-system
$ helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --namespace seldon-system \
    --set istio.enabled=true \
    --set certManager.enabled=true
{% endhighlight %} 

그외에 몇가지 옵션을 더 추가 할 수 있습니다.

 - `--set ambassador.enabled=true` :  Ambassador 사용시 해당 옵션을 추가 합니다. 
 - `--set istio.enabled=true` : [Istio](https://docs.seldon.io/projects/seldon-core/en/v1.1.0/ingress/istio.html) 사용시 해당 옵션을 추가 합니다.
 - `--set certManager.enabled=true` : [cert manager](https://cert-manager.io/docs/installation/kubernetes/)를 통해서 certificate 사용 
 
rollout 잘되었는지 확인해 봅니다. 

{% highlight bash %}
$ kubectl rollout status deploy/seldon-controller-manager -n seldon-system
deployment "seldon-controller-manager" successfully rolled out
{% endhighlight %} 

**Seldon Namespace 를 생성**

{% highlight bash %}
$ kubectl create namespace seldon
{% endhighlight %} 

## 1.3 Install Source to Image

Source to Image는 RedHat에서 지원하는 툴로서 코드를 빠르게 docker 로 만들어 줍니다. <br>
[https://github.com/openshift/source-to-image/releases](https://github.com/openshift/source-to-image/releases) 들어가서 다운로드 받습니다.


{% highlight bash %}
$ wget https://github.com/openshift/source-to-image/releases/download/v1.3.1/source-to-image-v1.3.1-a5a77147-linux-amd64.tar.gz
$ tar -xvf source-to-image-*.tar.gz
$ sudo cp ./s2i /usr/local/bin/
{% endhighlight %}


## 1.3 Ingress with Istio

설치는 istio.enabled=true 로 설치되어 있어야 합니다.

{% highlight yaml %}
$ cat <<EOF > gateway.yaml
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: seldon-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway # use istio default controller
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
EOF
{% endhighlight %}

{% highlight yaml %}
$ kubectl apply -f gateway.yaml
{% endhighlight %}



## 1.5 ~~Namespace Labeling~~  

<span style="color:red">  `serving.kubeflow.org/inferenceservice=enabled` 하면.. 
deploy시에 x509 certificate signed by unknown authority 에러 나옴 <br>
이유는 kubeflow의 apiserver 에 없는 뭔가를 호출하면서 발생.. 즉 kubeflow의 에러</span> 

~~아래 Namespace는 사용하고 있는 namespace 이름으로 변경하고,~~<br> 
~~`serving.kubeflow.org/inferenceservice=enabled` 로 label 을 지정합니다.~~

<div style="text-decoration: line-through">
{% highlight bash %}
$ kubectl create namespace seldon
$ kubectl label namespace seldon serving.kubeflow.org/inferenceservice=enabled
$ kubectl get ns seldon --show-labels
NAME     STATUS   AGE   LABELS
seldon   Active   26s   serving.kubeflow.org/inferenceservice=enabled
{% endhighlight %} 
</div>

# 2. Getting Started 

## 2.1 Pre-packaged Scikit-Learn Serving 



{% highlight yaml %}
$ cat <<EOF > sklearn.yaml 
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: sklearn
  namespace: seldon
spec:
  name: iris
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - name: classifier
          resources:
            requests:
              memory: 50Mi
    graph:
      children: []
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/sklearn/iris
      name: classifier
    name: anderson
    replicas: 1
EOF
{% endhighlight %} 

{% highlight yaml %}
$ kubectl apply -f sklearn.yaml
{% endhighlight %} 

확인은 다음과 같이 합니다.<br>
sdep는 seldondeployments 의 약자입니다.

{% highlight bash %}
$ kubectl get sdep -n seldon
NAME         AGE
iris-model   70m
{% endhighlight %}

Prediction Request는 다음과 같이 합니다.

{% highlight bash %}
$ INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
$ curl -X POST http://${INGRESS_HOST}/seldon/seldon/sklearn/api/v1.0/predictions \
     -H 'Content-Type: application/json' \
     -d '{ "data": { "ndarray": [[1,2,3,4]] } }' \
     --silent | jq
{
  "data": {
    "names": [
      "t:0",
      "t:1",
      "t:2"
    ],
    "ndarray": [
      [
        0.0006985194531162841,
        0.003668039039435755,
        0.9956334415074478
      ]
    ]
  },
  "meta": {}
}
{% endhighlight %}


Seldon External API를 통해서도 확인을 할 수 있습니다.<br>
아래 주소에서 반드시 끝에 슬래쉬까지 붙여줘야 합니다.

 - 형식: `http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc/`
 - 예제: http://{INGRESS_HOST}/seldon/seldon/sklearn/api/v1.0/doc/

<img src="{{ page.asset_path }}seldon-external-openapi.png" class="img-responsive img-rounded img-fluid" style="border: 2px solid #333333">


{% highlight bash %}
$ kubectl get svc istio-ingressgateway -n istio-system
NAME                   TYPE           CLUSTER-IP      EXTERNAL-IP                                 
istio-ingressgateway   LoadBalancer   10.100.223.12   abcd-12345.ap-northeast-2.elb.amazonaws.com 
{% endhighlight %}

**삭제**는 다음과 같이 합니다.

{% highlight bash %}
$ kubectl delete sdep iris-model -n seldon
$ kubectl delete deployments/iris-model-default-0-classifier -n seldon
{% endhighlight %}


## 2.2 Custom Model

아래 예제는 [예제](https://github.com/SeldonIO/seldon-core/tree/master/wrappers/s2i/python/test/model-template-app) 에서 가져왔습니다.<br>
파일이름은 MyModel.py 이고 클래스 이름도 동일하게 MyModel 로 가져갑니다.

{% highlight python %}
$ cat <<EOF > MyModel.py
import json
class MyModel:
    def __init__(self):
        print("MyModel 초기화됨 v0.0.11")

    def predict(self, data, names=None):
        print("predict함수 실행됨")
        print('data:', data, 'type:', type(data))
        print('names:', names, 'type:', type(names))
        return data
        
    def transform_input(self, X, names, meta=None):
        print('transform_input') 
        return X, names, meta
EOF
{% endhighlight %}
 

{% highlight python %}
$ cat <<EOF > requirements.txt
 scikit-learn
EOF
{% endhighlight %}

**Local 환경에서 테스트**를 합니다.

{% highlight bash %}
$ seldon-core-microservice MyModel REST --service-type MODEL
[2020-09-25 23:49:13 +0900] [16097] [INFO] Listening at: http://0.0.0.0:5000 (16097)

$ curl -X POST http://localhost:5000/api/v1.0/predictions \
     -H 'Content-Type: application/json' \
     -d '{"data": {"names":["f1", "f2"],  "ndarray": [[1,2,3,4], [1,2,3,4]]}}' --silent | jq
{% endhighlight %}

{% highlight json %}
{
  "jsonData": {
    "data": [
      [
        1,
        2,
        3,
        4
      ],
      [
        1,
        2,
        3,
        4
      ]
    ],
    "names": [
      "f1",
      "f2"
    ]
  },
  "meta": {}
}

{% endhighlight %}









**Docker build**가 필요합니다. <br>
`./s2i/environment` 라는 파일을 만들고 다음 내용을 넣습니다.

{% highlight bash %}
$ mkdir .s2i
$ cat <<EOF > .s2i/environment
MODEL_NAME=MyModel
API_TYPE=REST
SERVICE_TYPE=MODEL
PERSISTENCE=0
EOF
{% endhighlight %}  

{% highlight bash %}
$ s2i build . seldonio/seldon-core-s2i-python3 sklearn_iris:v0.0.12
$ docker tag sklearn_iris:v0.0.12 andersonjo/sklearn_iris:v0.0.12
$ docker push andersonjo/sklearn_iris:v0.0.12
{% endhighlight %}

**배포합니다.**

{% highlight yaml %}
$ kubectl apply -f - << END
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris-model
  namespace: seldon
spec:
  name: iris
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: andersonjo/sklearn_iris:v0.0.12
          resources:
            requests:
              memory: 50Mi
          env:
          - name: PAYLOAD_PASSTHROUGH
            value: "false"
          - name: SELDON_DEBUG
            value: "true"
    graph:
      name: classifier
    name: default
    replicas: 1
END
{% endhighlight %}

{% highlight yaml %}
$ INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
$ echo ${INGRESS_HOST}/seldon/seldon/iris-model/api/v1.0/doc/
{% endhighlight %}

아래 링크에서 확인합니다.<br>
[http://{INGRESS_HOST}.us-east-2.elb.amazonaws.com/seldon/seldon/iris-model/api/v1.0/doc/](http://{INGRESS_HOST}.us-east-2.elb.amazonaws.com/seldon/seldon/iris-model/api/v1.0/doc/)

<img src="{{ page.asset_path }}seldon-iris-model-extra-openapi.png" class="img-responsive img-rounded img-fluid" style="border: 2px solid #333333">


{% highlight bash %}
$ INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
$ curl -X POST http://${INGRESS_HOST}/seldon/seldon/iris-model/api/v1.0/predictions \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{"data": {"names":["f1", "f2"],  "ndarray": [[1,2,3,4], [1,2,3,4]]}}' --silent | jq
{
  "data": {
    "names": [
      "t:0",
      "t:1",
      "t:2"
    ],
    "ndarray": [
      [
        0.0006985194531162841,
        0.003668039039435755,
        0.9956334415074478
      ]
    ]
  },
  "meta": {}
}
{% endhighlight %}


**Python**에서 호출




{% highlight python %}
import requests
import json
test_url = 'http://****.us-east-2.elb.amazonaws.com/seldon/seldon/iris-model/api/v1.0/predictions'
jsonData = {
   "data": {
      "names": ["text"],
      "ndarray": [[1,2,3,4]]
   }
}
print(requests.post(test_url, data={'json': json.dumps(jsonData)}).text)
{% endhighlight %}







# Quick References

## 상태 확인 

**Seldon Models 리스트**

kubectl get sdep -n seldon


# Troubleshooting

## Pods 이 계속 Restarting 될때

{% highlight bash %}
$ kubectl get events -n seldon
{% endhighlight %}