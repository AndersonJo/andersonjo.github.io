---
layout: post 
title:  "ElasticSearch for Kafka Sink Connector"
date:   2022-04-01 01:00:00 
categories: "kafka"
asset_path: /assets/images/ 
tags: ['kafka', 'elasticsearch', 'connector']
---

# 1. Before you start

## 1.1 주요 링크

본문은 [ElasticSearch on EKS](/elasticsearch/2022/03/14/ElasticSearch-on-EKS/) 문서에서 나온 설치 방법 이후를 가정하고 있습니다.<br>
주요 링크는 아래와 같습니다.

- [ElasticSearch Sink Connector Documentation](https://docs.confluent.io/kafka-connect-elasticsearch/current/overview.html)를 확인 합니다.
- [ElasticSearch on EKS](/elasticsearch/2022/03/14/ElasticSearch-on-EKS/)


## 1.2 ES 연결 체크

먼저 ElasticSearch에 연결을 체크 합니다.

{% highlight bash %}
{% raw %}
# ES로 port-forward 연결을 합니다.              
$ kubectl port-forward -n elasticsearch service/elk-es-http 9200:9200

# 터미널 새로 열고, ES 접속 확인 합니다. 
$ PASSWORD=$(kubectl get secret elk-es-elastic-user -n elasticsearch -o go-template='{{.data.elastic | base64decode}}')
$ curl -u "elastic:${PASSWORD}" -k "http://localhost:9200" | jq
{
  "name": "elk-es-data-0",
  "cluster_name": "elk",
  <생략>
  "tagline": "You Know, for Search"
}
{% endraw %}
{% endhighlight %}

## 1.3 ES 권한 체크

ElasticSearch 에 create_index, read, write, 그리고 view_index_metadata 권한이 있어야 합니다. <br>
아래 코드는 권한 체크를 합니다.

{% highlight bash %}
{% raw %}
$ curl -XPOST -u "elastic:${PASSWORD}" -k "localhost:9200/_security/role/es_sink_connector_role?pretty" -H 'Content-Type: application/json' -d'
{
  "indices": [
    {
      "names": [ "*" ],
      "privileges": ["create_index", "read", "write", "view_index_metadata"]
    }
  ]
}'
{% endraw %}
{% endhighlight %}

## 1.4 Sink Connector 유저 생성

{% highlight bash %}
{% raw %}
$ export ES_SINK_PASSWORD=$(pwgen -n -c -y -s 25 1)
$ curl -XPOST -u "elastic:${PASSWORD}" -k "localhost:9200/_security/user/es_sink_connector_user?pretty" -H 'Content-Type: application/json' -d'
{
  "password" : "${ES_SINK_PASSWORD}",
  "roles" : [ "es_sink_connector_role" ]
}'
{% endraw %}
{% endhighlight %}

{% highlight bash %}
{
  "created" : true
}
{% endhighlight %}

유저 생성이 잘 됐는지 확인 합니다. 

{% highlight bash %}
$ curl -XGET -u "elastic:${PASSWORD}" -k "localhost:9200/_security/user/" | jq
{
  "es_sink_connector_user": {
    "username": "es_sink_connector_user",
    "roles": [
      "es_sink_connector_role"
    ],
    "full_name": null,
    "email": null,
    "metadata": {},
    "enabled": true
  }
}
{% endhighlight %}







# 2. Kafka Connector

## 2.1 Connector Properties


