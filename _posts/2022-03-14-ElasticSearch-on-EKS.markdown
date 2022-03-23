---
layout: post 
title:  "ElasticSearch on EKS"
date:   2022-03-14 01:00:00 
categories: "elasticsearch"
asset_path: /assets/images/ 
tags: []
---

# 1. Installation

## 1.1  Install ElasticSearch on Kubernetes

먼저 다음의 코드로 사전 준비를 합니다. 

{% highlight python %}
# 먼저 custom resource definitions 을 설치 합니다.
$ kubectl create -f https://download.elastic.co/downloads/eck/2.1.0/crds.yaml

# RBAC rule을 포함하는 operator를 설치합니다. 
$ kubectl apply -f https://download.elastic.co/downloads/eck/2.1.0/operator.yaml

# operator log를 확인합니다.
$ kubectl -n elastic-system logs -f statefulset.apps/elastic-operator
{% endhighlight %}


ElasticSearch는 최소 메모리 2GiB가 필요로 하며, 이보다 적을 경우 pod은 pending 상태에서 멈추게 됩니다.<br>
리소스 설정은 [링크](https://www.elastic.co/guide/en/cloud-on-k8s/current/k8s-managing-compute-resources.html#k8s-compute-resources) 를 참조 합니다.

{% highlight yaml %}
cat <<EOF > elasticsearch.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: elasticsearch
---
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: elk
  namespace: elasticsearch
spec:
  version: 8.1.0
  nodeSets:
  - name: master
    count: 1
    config:
      xpack.security.enrollment.enabled: true
      node.store.allow_mmap: false
      node.roles: ["master"]
    volumeClaimTemplates:
    - metadata:
        name: elasticsearch-data
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 10Gi
        storageClassName: gp2
  - name: data
    count: 2
    config:
      node.roles: ["data"]

    volumeClaimTemplates:
    - metadata:
        name: elasticsearch-data
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 512Gi
        storageClassName: gp2
    podTemplate:
      spec:
        containers:
        - name: elasticsearch
          env:
          - name: ES_JAVA_OPTS
            value: -Xms8g -Xmx8g
          resources:
            requests:
              memory: 8Gi
              cpu: 3.5
            limits:
              memory: 15Gi
        nodeSelector:
          node_role: ml-elasticsearch
  http:
    tls:
      selfSignedCertificate:
         disabled: true           # http service
---
apiVersion: kibana.k8s.elastic.co/v1
kind: Kibana
metadata:
  name: elk
  namespace: elasticsearch
spec:
  version: 8.1.1
  count: 1
  elasticsearchRef:
    name: elk
  podTemplate:
    spec:
      containers:
      - name: kibana
        env:
          - name: NODE_OPTIONS
            value: "--max-old-space-size=2048"
        resources:
          requests:
            memory: 2Gi
            cpu: 1
          limits:
            memory: 3.5Gi
            cpu: 2
  http:
    tls:
      selfSignedCertificate:
         disabled: true
EOF
{% endhighlight %}

설치후 확인 합니다.

{% highlight yaml %}
$ kubectl apply -f elasticsearch.yaml
$ kubectl get elasticsearch -n elasticsearch 
NAME               HEALTH   NODES   VERSION   PHASE   AGE
ml-elasticsearch   green    1       8.1.0     Ready   78m
{% endhighlight %}

## 1.2 Access Configuration

ES HTTP 서비스를 확인하고 port-forward를 걸어줍니다.

{% highlight yaml %}
$ kubectl get service -n elasticsearch elk-es-http
NAME          TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
elk-es-http   ClusterIP   172.20.177.28   <none>        9200/TCP   17m

# 포트 포워드 걸어주기
$ kubectl port-forward -n elasticsearch service/elk-es-http 9200:9200
{% endhighlight %}

이후 credentials을 얻고 access 요청을 보냅니다. 

{% highlight bash %}
{% raw %}
$ PASSWORD=$(kubectl get secret elk-es-elastic-user -n elasticsearch -o go-template='{{.data.elastic | base64decode}}')
$ curl -u "elastic:${PASSWORD}" -k "https://localhost:9200" | jq                                                            
{
  "name" : "elk-es-master-0",
  "cluster_name" : "elk",
  "cluster_uuid" : "abcdefghijklmnop_UUQQ",
  "version" : {
    "number" : "8.1.0",
    "build_flavor" : "default",
    "build_type" : "docker",
    "build_hash" : "1234567890abcdef01234567890abcd12345678a",
    "build_date" : "2022-03-03T14:20:00.690422633Z",
    "build_snapshot" : false,
    "lucene_version" : "9.0.0",
    "minimum_wire_compatibility_version" : "7.17.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "You Know, for Search"
}
{% endraw %}
{% endhighlight %}


## 1.3 Kibana

먼저 Kibana Pod 을 검색합니다. 

{% highlight bash %}
# 먼저 kibana 를 검색하고 해당 pod을 검색하기 위해서 selector를 검색합니다. 
$ kubectl get kibana -n elasticsearch -o yaml | grep selector
    selector: common.k8s.elastic.co/type=kibana,kibana.k8s.elastic.co/name=elk

# 해당 selector를 이용해서 kibana에 해당하는 pod 을 검색합니다.
$ kubectl get pod -n elasticsearch --selector='kibana.k8s.elastic.co/name=elk'
NAME                      READY   STATUS    RESTARTS   AGE
elk-kb-7f984c6b77-xgkfd   1/1     Running   0          9m41s
{% endhighlight %}

pod 이 정상적으로 떠있는 것을 확인했으면, Password를 얻고, service로 연결해서 들어갑니다. 

{% highlight bash %}
# Password 얻기
$ kubectl get secret elk-es-elastic-user -n elasticsearch -o=jsonpath='{.data.elastic}' | base64 --decode; echo
ABCDEFGHIJK12345678902K

# 연결
$ kubectl port-forward -n elasticsearch service/elk-kb-http 5601
{% endhighlight %}


[http://localhost:5601/](http://localhost:5601/) 로 접속을 합니다<br>
접속시 warning이 뜨는데 certificate authority 가 신뢰할수 없어서 그렇습니다. <br>
문제를 해결하기 위해서는 [링크](https://www.elastic.co/guide/en/cloud-on-k8s/current/k8s-tls-certificates.html#k8s-setting-up-your-own-certificate) 문서를 참조 합니다.

패스워드는 위에서 얻은 password를 사용하고 default username 은 **elastic** 입니다


<img src="{{ page.asset_path }}kibana-01.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">
