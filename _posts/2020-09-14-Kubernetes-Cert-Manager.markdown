---
layout: post
title:  "Cert Manager for Kubernetes"
date:   2020-09-14 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['kubernetes']
---

# 1. Install Cert Manager

## 1.1 Uninstall Cert-Manager

{% highlight bash %}
# 삭제 format 
$ kubectl delete -f https://github.com/jetstack/cert-manager/releases/download/vX.Y.Z/cert-manager.yaml

# 1.3.1 삭제
$ kubectl delete -f https://github.com/jetstack/cert-manager/releases/download/v1.3.1/cert-manager.yaml

# terminating state 로 멈춰 있다면..
$ kubectl delete apiservice v1beta1.webhook.cert-manager.io
{% endhighlight %}

## 1.2 Install Cert-Manager

 - 자세한 설치 방법은 [Kubernetes Installation](https://cert-manager.io/docs/installation/kubernetes/)를 참조 합니다.
 - Knative 설치시 cert-manger 1.3.1 까지만 지원이 되는듯 합니다. webhook 에러발생


{% highlight bash %}
$ kubectl version --short --client
Client Version: v1.19.3
{% endhighlight %}



{% highlight bash %}
# 일반적인 manifest 파일로 설치
# 만약 에러가 나면 apply를 replace 로 변경되 다시 apply 해볼 것
$ kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.3.1/cert-manager.yaml
{% endhighlight %}

설치 확인은 다음과 같이 하며, `cert-manager`, `cert-manager-cainjector`, `cert-manager-webhook` 이 있어야 합니다.

{% highlight bash %}
$ kubectl get pods --namespace cert-manager
NAME                                      READY   STATUS    RESTARTS   AGE
cert-manager-55658cdf68-7c5xw             1/1     Running   0          6m40s
cert-manager-cainjector-967788869-x6fgx   1/1     Running   0          6m40s
cert-manager-webhook-6668fbb57d-pbqd5     1/1     Running   0          6m40s
{% endhighlight %}


이후 Issuer 를 만드어서 webhook이 제대로 작동하는지 테스트해봅니다.<br>
`vi test-resources.yaml` 이후 다음을 작성합니다. 

{% highlight yaml %}
apiVersion: v1
kind: Namespace
metadata:
  name: cert-manager-test
---
apiVersion: cert-manager.io/v1beta2
kind: Issuer
metadata:
  name: test-selfsigned
  namespace: cert-manager-test
spec:
  selfSigned: {}
---
apiVersion: cert-manager.io/v1beta2
kind: Certificate
metadata:
  name: selfsigned-cert
  namespace: cert-manager-test
spec:
  dnsNames:
    - example.com
  secretName: selfsigned-cert-tls
  issuerRef:
    name: test-selfsigned
{% endhighlight %}


https://aws.amazon.com/blogs/containers/securing-kubernetes-applications-with-aws-app-mesh-and-cert-manager/