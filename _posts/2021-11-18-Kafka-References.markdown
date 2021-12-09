---
layout: post 
title:  "Apache Kafka References"
date:   2021-11-18 01:00:00 
categories: "data-engineering"
asset_path: /assets/images/ 
tags: []
---

<header>
    <img src="{{ page.asset_path }}kafka_background.jpeg" class="center img-responsive img-rounded img-fluid">
</header>


# 1. Kafka References 

## 1.1 Cluster Information

{% highlight bash %}
# AWS Kafka 의 ZookeeperConnectString 
$ aws kafka describe-cluster --region ap-northeast-2 --cluster-arn <CLUSTER_ARN> | jq .ClusterInfo.ZookeeperConnectString

# AWS Kafka Broker URL (Bootstrap Broker)
$ aws kafka get-bootstrap-brokers --region ap-northeast-2 --cluster-arn <CLUSTER_ARN> | jq
{% endhighlight %}

## 1.2 Kafka Console

일단 Kafka 가 잘 되는지 빠르게 확인하는 방법입니다. 

{% highlight bash %}
# Producer 생성 
$ kafka-console-producer.sh --topic "test-topic" --bootstrap-server localhost:9092

# Consumer 생성 
$ kafka-console-consumer.sh --topic "test-topic" --bootstrap-server localhost:9092
{% endhighlight %}
