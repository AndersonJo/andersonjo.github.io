---
layout: post
title:  "Seldon Core"
date:   2020-09-25 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['kubernetes']
---


# 1. Install Seldon Core with Kubeflow 

## 1.1 Install Seldon Core with Helm 

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
    --set istio.enabled=true 
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


## 1.2 Namespace Labeling  


아래 Namespace는 사용하고 있는 namespace 이름으로 변경하고,<br> 
`serving.kubeflow.org/inferenceservice=enabled` 로 label 을 지정합니다.

{% highlight bash %}
$ kubectl create namespace seldon
$ kubectl label namespace {Namespace} serving.kubeflow.org/inferenceservice=enabled
{% endhighlight %} 

## 1.5 Uninstall Seldon Core 

먼저 namespace 삭제해줍니다.

{% highlight bash %}
$ kubectl get ns
NAME                   STATUS   AGE
.. 생략
seldon-system          Active   4m34s

$ kubectl delete ns/seldon-system
{% endhighlight %}

Custom Resources 삭제는 다음과 같이 합니다. 

{% highlight bash %}
$ kubectl get crd
NAME                                                 CREATED AT
.. 생략
seldondeployments.machinelearning.seldon.io          2020-10-05T13:49:42Z

$ kubectl delete crd/seldondeployments.machinelearning.seldon.io
{% endhighlight %}
