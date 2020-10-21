---
layout: post
title:  "KFServing on EKS"
date:   2020-09-24 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['aws', 'machine-learning', 'ml-ops', 'seldon-core', 'mlops']
---

# 1. Setting Up!  

## 1.1 Serving namespace 지정

Kubeflow에서는 이미 KFServing 이 설치되어서 나옵니다.

{% highlight bash %}
$ kubectl create namespace kfserving
$ kubectl label namespace kfserving serving.kubeflow.org/inferenceservice=enabled
{% endhighlight %}

KFServing controller 가 설치되어 있는지 확인합니다.

{% highlight bash %}
$ kubectl get po -n kubeflow | grep kfserving-controller-manager
kfserving-controller-manager-0        2/2     Running   0          3h55m
{% endhighlight %}




## 1.2 Create Local Gateway

cat << EOF > ./local-cluster-gateway.yaml
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
spec:
  profile: empty
  components:
    ingressGateways:
      - name: cluster-local-gateway
        enabled: true
        label:
          istio: cluster-local-gateway
          app: cluster-local-gateway
        k8s:
          service:
            type: ClusterIP
            ports:
            - port: 15020
              name: status-port
            - port: 80
              name: http2
            - port: 443
              name: https
  values:
    gateways:
      istio-ingressgateway:
        debug: error
EOF 



## aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

{% highlight yaml %}
cat <<EOF > new-load-balancer.yaml
apiVersion: v1
kind: Service
metadata:
  name: kfserving-service
  namespace: kfserving
spec:
  type: LoadBalancer
  selector:
    app: nginx
  ports:
    - name: http
      protocol: TCP
      port: 80
      targetPort: 8080
    - name: https
      protocol: TCP
      port: 443
      targetPort: 8443
    - name: status-port
      port: 15021
      targetPort: 15021
    - name: tls
      port: 15443
      targetPort: 15443
EOF
{% endhighlight %}

## 1.2 Setting up Custom Ingress Gateway 

일단 Knative에서 새로운 Ingress Gateway 를 설정해줘야 합니다. 

{% highlight yaml %}
cat <<EOF > kfserving-ingress.yaml
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
spec:
  values:
    global:
      proxy:
        autoInject: disabled
      useMCP: false

  addonComponents:
    pilot:
      enabled: true
    prometheus:
      enabled: true

  components:
    ingressGateways:
      - name: ai-ingress-gateway
        enabled: true
        namespace: kfserving
        label:
          istio: ai-serving-gateway
      - name: ai-local-gateway
        enabled: true
        label:
          istio: ai-local-gateway
          app: ai-local-gateway
        k8s:
          service:
            type: LoadBalancer
            ports:
            - port: 15020
              name: status-port
            - port: 80
              name: http2
            - port: 443
              name: https
EOF
{% endhighlight %}


# 2. Getting Started

## 2.1 Iris Tutorial  

### 2.1.1 Model Deployment

{% highlight yaml %}
cat <<EOF > sklearn.yaml 
apiVersion: "serving.kubeflow.org/v1alpha2"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  default:
    predictor:
      sklearn:
        storageUri: "gs://kfserving-samples/models/sklearn/iris"
EOF
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
$ kubectl apply -f sklearn.yaml -n kfserving

# Service URL 을 확인합니다. (URL 뜨는데 까지 약 20~30초 걸림)
$ kubectl get inferenceservices sklearn-iris -n kfserving
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




### 2.1.2 Inference

 - INGRESS_HOST: Load Balancer Hostname (ex. `a0e3184ae3-1490218815.us-east-2.elb.amazonaws.com`)
 - INGRESS_PORT: Load Balancer Port (ex. `80`)
 - SERVICE_HOSTNAME: 모델 서빙되고 있는 주소 (ex. `sklearn-iris.kfserving.example.com`)

**환경변수 설정**

{% highlight bash %}
$ INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
$ INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
$ SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -n kfserving -o jsonpath='{.status.url}' | cut -d "/" -f 3)
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

ENV APP_HOME /app
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
$ k get inferenceservices -n kfserving
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


