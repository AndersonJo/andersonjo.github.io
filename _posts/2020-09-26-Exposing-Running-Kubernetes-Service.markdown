---
layout: post
title:  "Exposing Running Service on EKS"
date:   2020-09-26 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['aws']
---

# 1.Introduction 

EKS에서 3종류의 [Service Types](https://kubernetes.io/docs/concepts/services-networking/service/#publishing-services-service-types) 이 있습니다. 

 - **Cluster IP**: Service를 cluster-internal IP address로 노출합니다. (즉 Cluster 내부 IP)
 - **Node Port**: Node의 IP주소에 특정 port로 서비스를 노출합니다.  
 - **LoadBalancer**: Load balancer를 이용해서 서비스를 실제 외부로 노출 시킵니다.

Load Balancer를 사용해서 expose시키는 것은 EC2 nodes위에서 돌아가는 pods에 적용이 가능합니다.<br>
만약 AWS Fargate를 사용시 Load Balancer를 사용할수 없고, [ALB Ingress Controller](https://docs.aws.amazon.com/eks/latest/userguide/alb-ingress.html) 를 사용해야 합니다.

# 2. Nginx Tutorial 
 
## 2.1 Deployment 생성하기

`vi nginx-deployment.yaml` 로 deployment를 생성합니다.

{% highlight yaml %}
apiVersion: apps/v1 
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  replicas: 2 
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
{% endhighlight %}

{% highlight bash %}
$ kubectl apply -f nginx-deployment.yaml
$ kubectl get pods -l 'app=nginx' -o wide
NAME                                READY   STATUS    RESTARTS   AGE   IP              NODE                                       
nginx-deployment-85ff79dd56-ccz2r   1/1     Running   0          77s   192.168.0.251   ip-192-168-0-223.us-east-2.compute.internal
nginx-deployment-85ff79dd56-vvwcb   1/1     Running   0          77s   192.168.1.121   ip-192-168-1-192.us-east-2.compute.internal

{% endhighlight %}

## 2.2 Cluster IP Service

{% highlight yaml %}
cat <<EOF > clusterip.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  type: ClusterIP
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
EOF
{% endhighlight %}

{% highlight bash %}
$ kubectl apply -f clusterip.yaml
$ kubectl get service nginx-service
NAME            TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)   AGE
nginx-service   ClusterIP   10.100.202.157   <none>        80/TCP    57s

$ kubectl port-forward svc/nginx-service 5000:80
{% endhighlight %}

[http://localhost:5000](http://localhost:5000) 에서 확인합니다. 

## 2.3 NodePort Service 

{% highlight yaml %}
cat <<EOF > nodeport.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  type: NodePort
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
EOF
{% endhighlight %}

{% highlight bash %}
$ kubectl delete service nginx-service
$ kubectl apply -f nodeport.yaml
$ kubectl get svc
NAME            TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE
kubernetes      ClusterIP   10.100.0.1       <none>        443/TCP        4d18h
nginx-service   NodePort    10.100.220.236   <none>        80:32522/TCP   8s

$ kubectl port-forward svc/nginx-service 5000:80
{% endhighlight %}

[http://localhost:5000](http://localhost:5000) 에서 확인합니다. 

## 2.4 Load Balancer Service 

{% highlight yaml %}
cat <<EOF > loadbalancer.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  type: LoadBalancer
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
EOF
{% endhighlight %}


{% highlight bash %}
$ kubectl delete service nginx-service
$ kubectl apply -f loadbalancer.yaml
$ kubectl get svc
NAME            TYPE           CLUSTER-IP      EXTERNAL-IP                        PORT(S)        AGE
kubernetes      ClusterIP      10.100.0.1      <none>                             443/TCP        4d18h
nginx-service   LoadBalancer   10.100.84.201   ****.us-east-2.elb.amazonaws.com   80:32344/TCP   3s
{% endhighlight %}

확인은 다음과 같이 합니다.

{% highlight bash %}
curl ****.us-east-2.elb.amazonaws.com:80
{% endhighlight %}


<img src="{{ page.asset_path }}eks-nginx-example.png" class="img-responsive img-rounded img-fluid" style="border: 2px solid #333333">