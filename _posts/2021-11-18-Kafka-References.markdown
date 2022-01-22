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

## 1.1 Install Kafka

[Kafka Download](https://kafka.apache.org/downloads)에 들어가서 stable 버젼을 다운로드 받습니다.<br>
아래처럼 설치하되, anderson 부분은 수정을 합니다. 

{% highlight bash %}
# 3.0 인경우 
$ wget https://dlcdn.apache.org/kafka/3.0.0/kafka_2.13-3.0.0.tgz

# 설치
$ tar -xzf kafka_2.13-3.0.0.tgz
$ sudo mv kafka_2.13-3.0.0 /usr/local/kafka
$ sudo chown anderson:anderson -R /usr/local/kafka
{% endhighlight %}

.bashrc 를 열고 다음처럼 수정합니다.

{% highlight bash %}
$ export PATH=$PATH:/home/ubuntu/.local/bin/:/usr/local/kafka/bin
{% endhighlight %}


## 1.2 Cluster Information

{% highlight bash %}
$ export KAFKA_CLUSTER_ARN="<CLUSTER_ARN>"

# AWS Kafka 의 ZookeeperConnectString 
$ aws kafka describe-cluster --region ap-northeast-2 --cluster-arn $KAFKA_CLUSTER_ARN | jq .ClusterInfo.ZookeeperConnectString
$ export ZooKeeperConnect="<ZookeeperConnectString>"
$ export ZooKeeperConnect=$(aws kafka describe-cluster --cluster-arn $KAFKA_CLUSTER_ARN --output json | jq ".ClusterInfo.ZookeeperConnectString" | tr -d \")


# AWS Kafka Broker URL (Bootstrap Broker)
$ aws kafka get-bootstrap-brokers --region ap-northeast-2 --cluster-arn $KAFKA_CLUSTER_ARN | jq
$ export KafkaBootstrapConnect="<BootstrapBrokerString>"
$ export KafkaBootstrapConnect=$(aws kafka  get-bootstrap-brokers --cluster-arn $KAFKA_CLUSTER_ARN --output text)
{% endhighlight %}

## 1.3 Topic

{% highlight bash %}
# Topic 리스트
$ kafka-topics.sh --list --bootstrap-server $KafkaBootstrapConnect

# Topic 자세하게 보기 (Partition 정보 볼수 있음)
$ kafka-topics.sh --describe --bootstrap-server $ZooKeeperConnect
$ kafka-topics.sh --describe --bootstrap-server $ZooKeeperConnect --topic <TopicName>


# Topic 생성
$ kafka-topics.sh --bootstrap-server $ZooKeeperConnect --create --topic test-topic \
                  --partitions 3 --replication-factor 1
$ kafka-topics.sh --bootstrap-server $ZooKeeperConnect --create --topic test-topic \
                  --config min.insync.replicas=1 --partitions 10 --replication-factor 3


# Topic 삭제 
$ kafka-topics.sh --bootstrap-server $ZooKeeperConnect --delete --topic <TopicName>

# Retention 수정 (기본값은 7)
{% endhighlight %}

## 1.3 Kafka Console

일단 Kafka 가 잘 되는지 빠르게 확인하는 방법입니다. 

{% highlight bash %}
# Producer 생성 (--bootstrap-server localhost:9092)
$ kafka-console-producer.sh --topic "test-topic" --bootstrap-server $KafkaBootstrapConnect

# Consumer 생성 (--bootstrap-server localhost:9092)
$ kafka-console-consumer.sh --topic "test-topic" --bootstrap-server $KafkaBootstrapConnect
{% endhighlight %}

