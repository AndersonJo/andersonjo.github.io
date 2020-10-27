---
layout: post
title:  "Prometheus & Grafana on EKS (feat. Storage Class)"
date:   2020-09-08 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['aws', 'monitoring', 'gp2', 'storage']
---

# 1. Installation


## 1.1 Install Helm
 
**Helm을 설치**를 합니다.<br>

> tiller에 의존하던 건 2.x 버젼이고, helm init 같은 명령어는 필요없습니다. 

{% highlight bash %}
$ sudo snap install helm --classic
$ helm version --short
v3.3.4+ga61ce56
$ helm ls
{% endhighlight%}


## 1.2 Install Metrics Server

Metrics Server를 설치 합니다. <br>
자세한 설치방법은 [Metrics-Server Git Repository](https://github.com/kubernetes-sigs/metrics-server)을 참고 합니다.

{% highlight yaml %}
$ kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.3.7/components.yaml
{% endhighlight %}



## 1.3 Configure Storage 

AWS GP2 EBS volume을 사용해서 metrics 데이터를 저장합니다.<br>
[Storage Class 정보](https://kubernetes.io/docs/concepts/storage/storage-classes/)를 참조

각각의 StorageClass 는 `provisioner`, `parameters` 그리고 `reclaimPolicy` 3개의 필드가 존재합니다.<br>
AWS EBS 는 다음과 같이 사용할수 있습니다.

{% highlight yaml %}
cat <<EOF > prometheus-storageclass.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: prometheus
  namespace: prometheus
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp2
  fsType: ext4
reclaimPolicy: Retain
mountOptions:
  - debug
EOF
{% endhighlight%}

 - parameters
   - type: `io1`, `gp2`, `sc1`, `st1`.. 자세한건 [AWS EBS Volume Type](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-volume-types.html)을 참조
   - iopsPerGB: `io1` 에서만 사용가능 
   - fsType: Kubernetes 기본값은 ext4
 - encrypted: EBS volumne을 encrypted 할지 말지 설정. "true" 또는 "false" 로 설정
 - kmsKeyId: (Optional) encrypt 할때 사용되는 Key
 - allowVolumeExpansion: true
 
 
{% highlight bash %}
$ kubectl apply -f prometheus-storageclass.yaml 

{% endhighlight%}


## 1.4 Deploy Prometheus

{% highlight bash %}
$ helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
$ helm repo update
{% endhighlight%}

`helm search repo prometheus-community` 해당 명령어로 설치할 리스트를 볼 수 있습니다.



{% highlight bash %}
$ kubectl create namespace prometheus
$ helm install prometheus prometheus-community/prometheus \
    --namespace prometheus \
    --set alertmanager.persistentVolume.storageClass="gp2" \
    --set alertmanager.persistentVolume.size="512Gi" \
    --set server.persistentVolume.storageClass="gp2" \
    --set server.persistentVolume.size="512Gi"
{% endhighlight%}

설치가 잘 되었는지 확인해 봅니다.

{% highlight bash %}
$ kubectl get pods -n prometheus
NAME                                             READY   STATUS    RESTARTS   AGE
prometheus-alertmanager-86c44d7598-dkl99         2/2     Running   0          8m36s
prometheus-kube-state-metrics-6df5d44568-glfz4   1/1     Running   0          8m36s
prometheus-node-exporter-jm7jp                   1/1     Running   0          8m36s
prometheus-node-exporter-rgglp                   1/1     Running   0          8m36s
prometheus-node-exporter-vlk5n                   1/1     Running   0          8m36s
prometheus-node-exporter-z5bhj                   1/1     Running   0          8m36s
prometheus-pushgateway-6dfb58d9fb-bhb5v          1/1     Running   0          8m36s
prometheus-server-658677c9f5-qrcnk               2/2     Running   0          8m36s
{% endhighlight%}

PersistentVolume 도 체크 합니다.

{% highlight bash %}
$ kubectl get pv
NAME                                       CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                                STORAGECLASS   REASON   AGE
pvc-33e9f22e-fd00-41b3-b59a-a4649922c5eb   512Gi      RWO            Delete           Bound    prometheus/prometheus-alertmanager   gp2                     23s
pvc-e6136d3f-7c60-4918-82c5-b7b448869fcf   512Gi      RWO            Delete           Bound    prometheus/prometheus-server         gp2                     23s
{% endhighlight%}

<img src="{{ page.asset_path }}prometheus-pv.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

## 1.5 Connect Prometheus & AlertManager Server

prometheus server에 다음과 같이 접속 할 수 있습니다.

{% highlight bash %}
$ kubectl get ksvc helloworld-python -o jsonpath="{.status.url}"
$ POD_NAME=$(kubectl get pods --namespace prometheus -l "app=prometheus,component=server" -o jsonpath="{.items[0].metadata.name}")
$ kubectl --namespace prometheus port-forward $POD_NAME 9090
{% endhighlight%}

다른 방법으로 prometheus server에 접속하는 방법입니다. 

{% highlight bash %}
$ kubectl --namespace=prometheus port-forward deploy/prometheus-server 9090
{% endhighlight%}

[localhost:9090](localhost:9090) 으로 접속해서 확인합니다. 

<img src="{{ page.asset_path }}prometheus_example.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

AlertManager에 접속하는 방법은 다음과 같습니다.

{% highlight bash %}
$ export POD_NAME=$(kubectl get pods --namespace prometheus -l "app=prometheus,component=alertmanager" -o jsonpath="{.items[0].metadata.name}")
$ kubectl --namespace prometheus port-forward $POD_NAME 9093
{% endhighlight%}


## 1.6 Deploy Grafana 

먼저 DataSource yaml파일을 생성합니다. <br>
Grafana 가 어디에서 데이터를 가져와서 시각화해서 보여줄지 정의합니다.

{% highlight yaml %}
cat <<EOF > grafana.yaml
datasources:
  datasources.yaml:
    apiVersion: 1
    datasources:
    - name: Prometheus
      type: prometheus
      url: http://prometheus-server.prometheus.svc.cluster.local
      access: proxy
      isDefault: true
EOF
{% endhighlight%}

설치는 다음과 같이 합니다.

{% highlight bash %}
$ helm repo add grafana https://grafana.github.io/helm-charts
$ helm repo update
$ kubectl create namespace grafana
$ helm install grafana grafana/grafana \
    --namespace grafana \
    --set persistence.storageClassName="gp2" \
    --set persistence.size="512Gi" \
    --set persistence.enabled=true \
    --values grafana.yaml \
    --set service.type=LoadBalancer
{% endhighlight%}



암호및 접속 URL은 다음과 같이 정보를 얻을 수 있습니다.

{% highlight bash %}
# 암호 얻기 
$ kubectl get secret --namespace grafana grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
E13z4i5c7sD14rJ89gzrpjkSrvCnnSMzhO4hw7zen

$ kubectl get svc --namespace grafana grafana -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' ; echo 
ac276704341554e73a531964cf41ee40-304690984.us-east-2.elb.amazonaws.comn
{% endhighlight%}

위의 주소로 접속합니다.

<img src="{{ page.asset_path }}grafana_login.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">