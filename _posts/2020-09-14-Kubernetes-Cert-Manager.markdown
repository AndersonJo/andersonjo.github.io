---
layout: post
title:  "Cert Manager for Kubernetes"
date:   2020-09-14 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['kubernetes']
---

# 1. Install Cert Manager

## 3.6 Install Cert-Manager

 - 자세한 설치 방법은 [Kubernetes Installation](https://cert-manager.io/docs/installation/kubernetes/)를 참조 합니다.

먼저 kubectl client version을 확인합니다.<br>
**반드시 v1.19.0-rc.1 보다 높아야 하며**, 낮을 경우 CRD 업데이트에서 에러가 생깁니다.
SSL 101
{% highlight bash %}
$ kubectl version --short --client
Client Version: v1.19.3
{% endhighlight %}

{% highlight bash %}
# 일반적인 manifest 파일로 설치
$ kubectl apply --validate=false -f https://github.com/jetstack/cert-manager/releases/download/v1.0.2/cert-manager.yaml
{% endhighlight %}

설치 확인은 다음과 같이 하며, `cert-manager`, `cert-manager-cainjector`, `cert-manager-webhook` 이 있어야 합니다.

{% highlight bash %}
$ kubectl get pods --namespace cert-manager
NAME                                       READY   STATUS    RESTARTS   AGE
cert-manager-cainjector-774bd85548-w2vkb   1/1     Running   0          2d4h
cert-manager-f8f6c65f9-7klrk               1/1     Running   0          2d4h
cert-manager-webhook-59fb566ff-29wfj       1/1     Running   0          2d4h
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