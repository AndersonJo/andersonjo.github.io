---
layout: post
title:  "Kubernetes Quick References"
date:   2019-05-11 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['pod', 'docker']
---



<header>
    <img src="{{ page.asset_path }}kubernetes-wallpaper.jpg" class="img-responsive img-rounded img-fluid">
    <div style="text-align:right;">
    <a style="background-color:black;color:white;text-decoration:none;padding:4px 6px;font-family:-apple-system, BlinkMacSystemFont, &quot;San Francisco&quot;, &quot;Helvetica Neue&quot;, Helvetica, Ubuntu, Roboto, Noto, &quot;Segoe UI&quot;, Arial, sans-serif;font-size:12px;font-weight:bold;line-height:1.2;display:inline-block;border-radius:3px" href="https://unsplash.com/@chuttersnap?utm_medium=referral&amp;utm_campaign=photographer-credit&amp;utm_content=creditBadge" target="_blank" rel="noopener noreferrer" title="Download free do whatever you want high-resolution photos from chuttersnap"><span style="display:inline-block;padding:2px 3px"><svg xmlns="http://www.w3.org/2000/svg" style="height:12px;width:auto;position:relative;vertical-align:middle;top:-2px;fill:white" viewBox="0 0 32 32"><title>unsplash-logo</title><path d="M10 9V0h12v9H10zm12 5h10v18H0V14h10v9h12v-9z"></path></svg></span><span style="display:inline-block;padding:2px 3px">chuttersnap</span></a> 
    </div>
</header>




# Quick References


## Version Check

Server 그리고 client version 둘다 확인 할 수 있습니다. 

{% highlight bash %}
$ kubectl version
{% endhighlight %}

## Context

context 정보는 다음과 같이 얻습니다. 

{% highlight bash %}
$ kubectl config get-contexts
          anderson-context   anderson-cluster   anderson.jo   
*         minikube           minikube           minikube   
{% endhighlight %}

다른 context로 변경은 다음과 같이 합니다. 

{% highlight bash %}
$ kubectl config use-context anderson-context
Switched to context "anderson-context".
{% endhighlight %}

## Cluster Information

{% highlight bash %}
$ kubectl cluster-info
Kubernetes master is running at https://172.17.0.16:8443
KubeDNS is running at https://172.17.0.16:8443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
{% endhighlight %}


위의 명령어에서 master 그리고 Kubernetes DNS서비스 돌아가는 것을 확인 할 수 있습니다. 

- **Kubernetes master**: master
- **KubeDNS**: DNS
- **kubernetes-dashboard**: dashboard - UI에서 applications을 확인 가능



## All Nodes Information

applications을 host시킬 수 있는 모든 nodes 정보를 보여줍니다. 

{% highlight bash %}
$ kubectl get nodes
NAME       STATUS   ROLES    AGE   VERSION
minikube   Ready    master   12m   v1.13.3
{% endhighlight %}

## Namespace

namespace 정보는 다음과 같이 얻습니다. 

{% highlight bash %}
$ kubectl get namespaces
NAME              STATUS   AGE
default           Active   8d
kube-node-lease   Active   8d
kube-public       Active   8d
kube-system       Active   8d
ml-server         Active   2s
{% endhighlight %}

Namespace 생성은 다음과 같이 합니다 .

{% highlight bash %}
$ kubectl create namespace ml-server
namespace/ml-server created
{% endhighlight %}

## Pods

{% highlight bash %}
$ kubectl get pod -n ml-server
NAME                                             READY   STATUS    RESTARTS   AGE
misc-mobdata-ml-server-deploy-6b5748b65c-9wpbk   0/1     Pending   0          33s
{% endhighlight %}

정확하게  app을 지정해서 볼 수 도 있습니다. 

{% highlight bash %}
$ kubectl get pod -n ml-server -l app=ml-app
{% endhighlight %}

Pod에 대한 정보는 다음의 명령어로 알 수 있습니다.

{% highlight bash %}
$ kubectl describe pod misc-mobdata-ml-server-deploy-6b5748b65c-9wpbk -n ml-server
{% endhighlight %}


## Deploy

{% highlight bash %}
$ kubectl apply -f deploy.yaml
{% endhighlight %}



# Port Forward for Testing

### Port 확인

**Port를 확인**합니다.

{% highlight bash %}
$ kubectl get pods ml-server-deploy-6b4b98486-k4srw -n alpha 
NAME                               READY   STATUS    RESTARTS   AGE
ml-server-deploy-6b4b98486-k4srw   1/1     Running   0          59m

$ kubectl get pods ml-server-deploy-6b4b98486-k4srw -n alpha --template='{{(index (index .spec.containers 0).ports 0).containerPort}}{{"\n"}}'
80
{% endhighlight %}

**port-forward 를 사용해서 local port를 pod에 있는 port로 연결을 시킵니다.**

resource name을 지정할수 있는데 deployment, service , pod 등을 선택할 수 있습니다. 

5000:80 에서 앞쪽 포트가 localhost에서 사용할 포트이고, 80이 pod의 접속 포트 입니다.



### Pods으로 접속

먼저 **Pods 정보**를 확인하고 ready인지 확인합니다. 

{% highlight bash %}
$ kubectl get pods  -n alpha -l app=ml-app
NAME                               READY   STATUS    RESTARTS   AGE
ml-server-deploy-6b4b98486-k4srw   1/1     Running   0          56m
{% endhighlight %}

**Pods으로 접속**

{% highlight bash %}
$ kubectl port-forward ml-server-deploy-6b4b98486-k4srw -n alpha 5000:80
Forwarding from 127.0.0.1:5000 -> 80
Forwarding from [::1]:5000 -> 80
{% endhighlight %}

**pods/ 를 추가**해도 됨

{% highlight bash %}
$ kubectl port-forward pods/ml-server-deploy-6b4b98486-k4srw -n alpha 5000:80
Forwarding from 127.0.0.1:5000 -> 80
Forwarding from [::1]:5000 -> 80
{% endhighlight %}

### Deployment로 접속

**Deployments 상태**도 확인을 합니다.

{% highlight bash %}
$ kubectl get deployment -n alpha -l app=ml-app
NAME               DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
ml-server-deploy   1         1         1            1           7h
{% endhighlight %}

**deployment/** 를 붙여서 접속

{% highlight bash %}
$ kubectl port-forward deployment/ml-server-deploy -n alpha 5000:80
Forwarding from 127.0.0.1:5000 -> 80
Forwarding from [::1]:5000 -> 80
Handling connection for 5000
{% endhighlight %}



### ReplicaSet 으로 접속

**ReplicaSet**도 상태를 확인합니다.

{% highlight bash %}
$ kubectl get rs -n alpha -l app=ml-app
NAME                          DESIRED   CURRENT   READY   AGE
ml-server-deploy-6cd9bdc5d4   1         1         1       1h
{% endhighlight %}

**ReplicaSet으로 접속**합니다. 

{% highlight bash %}
$ kubectl port-forward rs/ml-server-deploy-6cd9bdc5d4 -n alpha 5000:80
Forwarding from 127.0.0.1:5000 -> 80
Forwarding from [::1]:5000 -> 80
Handling connection for 5000
{% endhighlight %}



### Service로 접속

**서비스**를 확인합니다.

{% highlight bash %}
$ kubectl get svc  -n alpha -l app=ml-app
NAME            TYPE       CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
ml-server-svc   NodePort   10.231.46.95   <none>        80:30021/TCP   3d
{% endhighlight %}

서비스로 접속을 합니다.

{% highlight bash %}
$ kubectl port-forward svc/ml-server-svc -n alpha 5000:80
Forwarding from 127.0.0.1:5000 -> 80
Forwarding from [::1]:5000 -> 80
Handling connection for 5000
{% endhighlight %}





















# Minikube

Minikube는 Kubernetes를 local 환경에서 쉽게 돌릴수 있는 툴입니다. 

Minikube는 single-node Kuebernetes cluster이며 VM위에서 돌아가며, 

local 개발시에 편리하게 사용할 수 있습니다. 



[Minikube 설치하기](https://kubernetes.io/docs/tasks/tools/install-minikube)

실행은 다음과 같이 합니다. 

{% highlight bash %}
sudo minikube start
{% endhighlight %}



Minikube를 실행하기 위해서는 VM이 설치되어 있어야  합니다. 

하지만 저처럼 귀찮니즘에 쩔어있는 개발자라면 VM없이 실행시킬수 있는 방법이 있습니다. 

아래의 옵션을 더 추가해서 실행을 시킵니다.

{% highlight bash %}
sudo minikube start --vm-driver=none --apiserver-ips 127.0.0.1 --apiserver-name localhost
{% endhighlight %}



만약에 권한과 관련된 warning이 뜨면 다음의 명령어로 권한을 수정해 줍니다.

(개인적으로는 해줄 필요가 없었음)

{% highlight bash %}
sudo chown -R $USER $HOME/.minikube
sudo chgrp -R $USER $HOME/.minikube
sudo chown -R $USER $HOME/.kube
sudo chgrp -R $USER $HOME/.kube
{% endhighlight %}

또는 delete 명령어도 도움이 될 수 있습니다. 

{% highlight bash %}
sudo minikube delete
{% endhighlight %}



