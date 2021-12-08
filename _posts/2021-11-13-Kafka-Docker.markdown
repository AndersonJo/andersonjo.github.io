---
layout: post 
title:  "Apache Kafka with Docker"
date:   2021-11-15 01:00:00 
categories: "data-engineering"
asset_path: /assets/images/ 
tags: []
---

<header>
    <img src="{{ page.asset_path }}kafka_background.jpeg" class="center img-responsive img-rounded img-fluid">
</header>

# 1. Installation

## 1.1 Docker Compose Installation

[https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/) 를 참고합니다. 

{% highlight bash %}
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
$ docker-compose --version
docker-compose version 1.29.2, build 5becea4c
{% endhighlight %}


## 1.2 docker-compose.yml

`KAFKA_ADVERTISED_HOST_NAME` 를 host ip 로 설정합니다. <br>
이때 localhost 또는 127.0.0.1 같은 도메인으로 잡으면, multiple brokers를 실행할 수 없게 됩니다.<br>
하지만 multi brokers를 생성하지 않고 그냥 개발용으로 쓸 것이니 그냥 127.0.0.1 을 사용합니다. 

- [docker-compose.yml](https://raw.githubusercontent.com/wurstmeister/kafka-docker/master/docker-compose.yml)

{% highlight yaml %}
version: '2'
services:
  zookeeper:
    image: wurstmeister/zookeeper
    container_name: zookeeper
    ports:
      - "2181:2181"
  kafka:
    image: wurstmeister/kafka
    container_name: kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_HOST_NAME: 127.0.0.1
      KAFKA_ADVERTISED_PORT: 9092
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
{% endhighlight %}


{% highlight bash %}
$ docker-compose -f docker-compose.yml up -d
{% endhighlight %}

## 1.3 Creating Topic

먼저 kafka container 에서 shell을 실행시켜서, 이 안에서 topic을 만들 수 있습니다.

{% highlight bash %}
$ docker exec -it kafka /bin/sh
$ cd /opt/kafka/bin/

# topic 생성
$ kafka-topics.sh --create \
                  --zookeeper zookeeper:2181 \
                  --replication-factor 1 \
                  --partitions 1 \
                  --topic test_topic

# 확인합니다.
$ kafka-topics.sh --list --zookeeper zookeeper:2181
{% endhighlight %}

## 1.4 Example 

**producer.py**

{% highlight python %}
import json
from kafka import KafkaProducer

def produce():
    producer = KafkaProducer(
        acks=0,
        bootstrap_servers=[
            'localhost:9092',
        ],
        api_version=(2, 8, 1),
        value_serializer=lambda x: json.dumps(x).encode('utf-8'))

    for i in range(10):
        print(f'Sending: {i}')
        r = producer.send('TestTopic', value=f'test {i}')
        producer.flush()

if __name__ == '__main__':
    produce()
{% endhighlight %}

**consumer.py** 

{% highlight python %}
import json
from kafka import KafkaConsumer

def consume():
    consumer = KafkaConsumer(
        'TestTopic',
        bootstrap_servers=[
            'localhost:9092',
        ],
        api_version=(2, 8, 1),
        auto_offset_reset='earliest',
        group_id='my-group',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.dumps(x.decode('utf-8')))

    for message in consumer:
        print(message)

if __name__ == '__main__':
    consume()
{% endhighlight %}