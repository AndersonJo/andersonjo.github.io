---
layout: post
title:  "EKS Deployment Example"
date:   2020-09-06 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: []
---

# Prerequisite

해당 문서는 [Amazon ECR](https://incredible.ai/aws/2020/08/14/Amazon-ECR/)을 수행한 이후입니다. <br>
다만 image 부분에 container 갈아치우는 부분만 수정하면.. ECR없이도 돌아갈수 있도록 설명해놨습니다. 


# 2. Kubernetes 

## 2.1 Deployment 

 - spec.template.spec.containers.image <- ECS container 주소로 바꾸면 ECS에서 pull받게 됩니다. <br>
   - 예를 들어서 아래와 같이 변경하면 됩니다.<br>  
     `image: 826443632289.dkr.ecr.us-east-1.amazonaws.com/jenkins-test:v0.0.1`

{% highlight bash %}
cat <<EOF > nginx-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2  # ECS Container 로 수정
        imagePullPolicy: Always
        ports:
        - containerPort: 80
EOF
{% endhighlight %}

{% highlight bash %}
$ kubectl apply -f nginx-deployment.yaml
$ kubectl get pods
NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-5bb5f7b647-lh5hh   1/1     Running   0          2m7s
nginx-deployment-5bb5f7b647-v26jv   1/1     Running   0          55s
{% endhighlight %}





## 2.2 LoadBalancer Service 

{% highlight bash %}
cat <<EOF > loadbalancer.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service-loadbalancer
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
$ kubectl create -f loadbalancer.yaml
NAME                         TYPE           CLUSTER-IP       EXTERNAL-IP                                   PORT(S)        AGE
kubernetes                   ClusterIP      10.100.0.1       <none>                                        443/TCP        4h19m
nginx-service-loadbalancer   LoadBalancer   10.100.160.221   a171c1bb-129660.us-east-1.elb.amazonaws.com   80:31063/TCP   16s
{% endhighlight %}

테스트
{% highlight bash %}
$ curl a171c1bb-129660.us-east-1.elb.amazonaws.com:80 | grep title
{% endhighlight %}