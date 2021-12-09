---
layout: post 
title:  "Apache Kafka ETC"
date:   2021-11-18 01:00:00 
categories: "data-engineering"
asset_path: /assets/images/ 
tags: []
---

<header>
    <img src="{{ page.asset_path }}kafka_background.jpeg" class="center img-responsive img-rounded img-fluid">
</header>


# 1. Kafka ETC 

## 1.1 Kafka Console

일단 Kafka 가 잘 되는지 빠르게 확인하는 방법입니다. 

{% highlight bash %}
# Producer 생성 
$ kafka-console-producer.sh --topic "test-topic" --bootstrap-server localhost:9092

# Consumer 생성 
$ kafka-console-consumer.sh --topic "test-topic" --bootstrap-server localhost:9092
{% endhighlight %}
