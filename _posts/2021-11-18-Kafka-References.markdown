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


# 1. Installation 

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


## 1.2 Install Confluent Kafka

[Confluent 설치 문서](https://docs.confluent.io/platform/current/installation/installing_cp/deb-ubuntu.html) 참조합니다.

{% highlight bash %}
$ wget -qO - https://packages.confluent.io/deb/7.0/archive.key | sudo apt-key add -
$ sudo add-apt-repository "deb [arch=amd64] https://packages.confluent.io/deb/7.0 stable main"
$ sudo add-apt-repository "deb https://packages.confluent.io/clients/deb $(lsb_release -cs) main"
$ sudo apt-get update && \
    sudo apt-get install confluent-platform && \
    sudo apt-get install confluent-security
$ sudo apt-get update && sudo apt-get install confluent-community-2.13
{% endhighlight %}


## 1.3 Kafka-UI

[Installation Document](https://github.com/provectus/kafka-ui/blob/master/charts/kafka-ui/README.md)<br>
중요한건 ZOOKEEPER 또는 BOOTSTRAPSERVERS 에 값을 넣을때, 모든값을 넣을 필요는 없고, 브로커 1개만 넣으면 충분함.

{% highlight bash %}
# 재시작시 
$ docker start kafka-ui

# 처음 시작시
$ docker run -p 8070:8080 \
    --name kafka-ui \
	-e KAFKA_CLUSTERS_0_NAME=kafka-ml \
	-e KAFKA_CLUSTERS_0_ZOOKEEPER=z-1.kafka-dev.a123456.a1.kafka.ap-northeast-2.amazonaws.com:2181 \
	-e KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS=b-1.kafka-dev.a123456.a1.kafka.ap-northeast-2.amazonaws.com:9092 \
	-d provectuslabs/kafka-ui:latest
{% endhighlight %}



# 2. Kafka References

## 2.2 Cluster Information

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

## 2.3 Topic

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

## 2.4 Kafka Console

일단 Kafka 가 잘 되는지 빠르게 확인하는 방법입니다. 

{% highlight bash %}
# Producer 생성 (--bootstrap-server localhost:9092)
$ kafka-console-producer.sh --topic "test-topic" --bootstrap-server $KafkaBootstrapConnect

# Consumer 생성 (--bootstrap-server localhost:9092)
$ kafka-console-consumer.sh --topic "test-topic" --bootstrap-server $KafkaBootstrapConnect
{% endhighlight %}

