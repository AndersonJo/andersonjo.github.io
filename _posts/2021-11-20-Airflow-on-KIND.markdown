---
layout: post 
title:  "Airflow on KIND"
date:   2021-11-20 01:00:00 
categories: "airflow"
asset_path: /assets/images/ 
tags: []
---

<header>
    <img src="{{ page.asset_path }}airflow_01.png" class="center img-responsive img-rounded img-fluid">
</header>



# 1. Introduction

공식문서는 [Hell Chart for Airflow](https://airflow.apache.org/docs/helm-chart/stable/index.html)를 참고 합니다.

제 글에서는 다음의 것들을 할 것입니다. 

1. Kind 설치
2. Helm 을 사용해서 Airflow 를 Kind 그리고 AWS EKS에 배포해보도록 하겠습니다.


# 2. Kind

## 2.1 Kind Installation 

[https://kind.sigs.k8s.io/docs/user/quick-start/](https://kind.sigs.k8s.io/docs/user/quick-start/)에서 설치 방법이 있습니다.

Ubuntu의 경우는 다음과 같이 설치 합니다. 

{% highlight bash %}
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.11.1/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
{% endhighlight %}

## 2.2 Kind Cluster

Latest version의 Kubernetes로 클러스터를 생성합니다.<br> 
1.21.1 을 선택한 이유는 일단 현재 시점에서 EKS Kubernetes Version과 최대한 동일하게 가려고 하기 위함입니다. 

{% highlight bash %}
$ kind create cluster --image kindest/node:v1.21.1

# 설치 잘 됐는지 확인
$ kubectl cluster-info --context kind-kind
{% endhighlight %}

## 2.3 Install Airflow

{% highlight bash %}
# Airflow Helm Stable Repo를 추가시킵니다. 
$ helm repo add apache-airflow https://airflow.apache.org
$ helm repo update

# Namespace 를 설정합니다. 
$ export NAMESPACE=airflow
$ kubectl create namespace $NAMESPACE
$ kubectl get namespaces airflow
{% endhighlight %}

Airflow를 설치 합니다.

{% highlight bash %}
# 만약 Example DAGs 까지 모두 설치하고자 한다면 아래 옵션을 더 추가 시켜 줍니다.  
# --set 'env[0].name=AIRFLOW__CORE__LOAD_EXAMPLES,env[0].value=True'
$ export RELEASE_NAME=airflow
$ helm install $RELEASE_NAME apache-airflow/airflow --namespace $NAMESPACE
{% endhighlight %}

설치 확인을 합니다. 

{% highlight bash %}
$ kubectl get pods --namespace airflow
$ helm list --namespace airflow
NAME   	NAMESPACE	REVISION	UPDATED                                	STATUS  	CHART        	APP VERSION
airflow	airflow  	1       	2021-12-02 13:13:09.877693925 +0900 KST	deployed	airflow-1.3.0	2.2.1 
{% endhighlight %}




## 2.4 주요 접속 경로 

설치를 다하게 되면 위에 있는 것처럼 Airflow에 접속할 수 있습니다.

- Webserver: `kubectl port-forward svc/airflow-webserver 8080:8080 --namespace airflow`
  - default Username: `admin`
  - default Password: `admin`
- Postgre Connection
  - default Username: `postgres`
  - default Password: `postgres`
  - Port: `5432`
- Dashboard: `kubectl port-forward svc/airflow-flower 5555:5555 --namespace airflow`

- **Secret Key**는 다음과 같이 얻습니다.

{% highlight bash %}
$ echo Fernet Key: $(kubectl get secret --namespace airflow airflow-fernet-key -o jsonpath="{.data.fernet-key}" | base64 --decode)
{% endhighlight %}

**웹서버 접속**은 다음과 같이 합니다. 

{% highlight bash %}
$ kubectl port-forward svc/airflow-webserver 8080:8080 --namespace airflow
{% endhighlight %}



# 3. Custom Airflow Image 

## 3.1 Adding DAGs

dags 디렉토리에 DAG파일을 추가시키면 됩니다.

{% highlight bash %}
mkdir my-airflow-project && cd my-airflow-project
mkdir dags  # put dags here

cat <<EOM > Dockerfile
FROM apache/airflow
COPY . .
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends vim 
USER airflow
EOM
{% endhighlight %}

Docker Build 시키고 배포합니다. 

{% highlight bash %}
$ docker build --tag my-dags:0.0.1 .
$ kind load docker-image my-dags:0.0.1
$ helm upgrade airflow apache-airflow/airflow --namespace airflow \
    --set images.airflow.repository=my-dags \
    --set images.airflow.tag=0.0.1
{% endhighlight %}









