---
layout: post 
title:  "Apache Kafka & Connector"
date:   2021-11-17 01:00:00 
categories: "data-engineering"
asset_path: /assets/images/ 
tags: []
---

<header>
    <img src="{{ page.asset_path }}kafka_background.jpeg" class="center img-responsive img-rounded img-fluid">
</header>

# 1. Installation


## 1.1 Create a Topic on Docker

{% highlight bash %}
$ docker exec -it kafka /bin/sh
$ cd /opt/kafka/bin/
$ kafka-topics.sh --create \
                  --zookeeper zookeeper:2181 \
                  --replication-factor 1 \
                  --partitions 1 \
                  --topic "test-topic"
$ kafka-topics.sh --list --zookeeper zookeeper:2181
{% endhighlight %}



## 1.2 Installing Confluent Connectors (Optional)

먼저 connector는 [confluent](https://www.confluent.io/hub/) 를 통해서 검색할수 있습니다. <br>
Confluent 를 통해서는 commercial 서비스를 받을 수 있고, 기본적으로 제공되는 plugins 외에 다양한 connectors 를 설치 할 수 있습니다.

다운로드 받을 connectors는 다음과 같습니다.<br> 
`Download` 버튼을 눌러서 직접 다운로드 받고 설치 하면 됩니다.<br> 

1. 다운로드 받습니다.
2. unzip 시킵니다. 
3. /usr/local/kafka/plugins/<new connector>  위치로 lib 디렉토리를 이동시킵니다.  
4. 이후  `plugin.path=/usr/local/kafka/plugins` 를 설정파일에 추가하면 됩니다.

중요한건 lib 디렉토리안에 들어있는 jar 파일들이고<br>
**/usr/local/kafka/plugins 아래에 있는 디렉토리들은 바로 jar 파일들이 존재해야 합니다.**<br>
즉 /usr/local/kafka/plugins/new-connector/lib/aaa.jar 이런식으로 존재하면 안됩니다.




# 2. Running Connector

## 2.1 Standalone 

**connect-file-source.properties** 파일 생성

 - `connector.class=FileStreamSource` : 파일 커넥터 클래스를 사용한다고 정의
 - `topic=test-topic` : 보낼 Topic 이름을 지정
 - FileStreamSource 뒷쪽에 스페이스가 존재할경우 에러가 날수 있습니다. ㅠㅠ 4시간 날려먹음

{% highlight yaml %}
name=local-file-source
connector.class=FileStreamSource
tasks.max=1
file=/home/anderson/Downloads/test.txt
topic=test-topic
{% endhighlight %}


**connect-file-sink.properties** 

{% highlight yaml %}
name=local-file-sink  
connector.class=FileStreamSink
tasks.max=1
file=/home/anderson/Downloads/test.sink.txt  
topics=test-topic
{% endhighlight %}


**connect-standalone.properties**

{% highlight yaml %}
bootstrap.servers=localhost:9092
key.converter=org.apache.kafka.connect.json.JsonConverter
value.converter=org.apache.kafka.connect.json.JsonConverter
key.converter.schemas.enable=true
value.converter.schemas.enable=true
offset.storage.file.filename=/tmp/connect-file.offsets

# 아래설정을 해주면 기본값보다 더 빠르게 메세지를 보냅니다. 
offset.flush.timeout.ms=1000
buffer.memory=100
plugin.path=/usr/local/kafka/plugins
{% endhighlight %}

또는 StringConverter 도 사용할수 있습니다. <br>
위의 코드를 대체하면 됩니다.

{% highlight yaml %}
key.converter=org.apache.kafka.connect.storage.StringConverter  
value.converter=org.apache.kafka.connect.storage.StringConverter  
{% endhighlight %}





**Standalone 모드 실행**

{% highlight bash %}
$ connect-standalone.sh connect-standalone.properties connect-file-source.properties connect-file-sink.properties
{% endhighlight %}

터미널을 새로 열고.. 

{% highlight bash %}
$ tail -f test.sink.txt
{% endhighlight %}

또 새로운 터미널을 열고 테스트를 해봅니다.

{% highlight bash %}
$ echo Hello >> test.txt
$ echo "I am Anderson" >> test.txt
$ echo "I am a boy you are a girl" >> test.txt
$ echo '{"hello": 123.4, "key": "Stock Market"}' >> test.txt
{% endhighlight %}



## 2.2 Distributed Mode

Distributed Mode 에서는 여러개의 workers가 동일한 `group.id` 로 Kafka Connect를 실행하게 됩니다.<br>
또한 offset, task configuration, status 등을 topic 에 저장해놓기 때문에 이것도 standalone 과 다릅니다. 

**connect-distributed.properties**

{% highlight yaml %}
bootstrap.servers=localhost:9092

# unique name for the cluster
# consumer group ID와 중복되면 안됨
group.id=connect-cluster

key.converter=org.apache.kafka.connect.json.JsonConverter
value.converter=org.apache.kafka.connect.json.JsonConverter
key.converter.schemas.enable=true
value.converter.schemas.enable=true

# offsets을 저장할 topic / the topic should have many partitions and be replicated and compacted. 
# 해당 토픽은 자동으로 생성됨 
# default replication factor = 3 이며 특별한 케이스에서는 더 커질수도 있다
# 개발환경에서는 single-broker 로 움직이기 때문에 1로 설정
offset.storage.topic=connect-offsets
offset.storage.replication.factor=1
#offset.storage.partitions=25

# connector 그리고 task configuration을 저장할 Topic을 설정합니다. 
# this should be a single partition, highly replicated and compacted topic. 
# 해당 토픽은 자동으로 생성됨
# default replication factor = 3 이며 특별한 케이스에서는 더 커질수도 있다
# 개발환경에서는 single-broker 로 움직이기 때문에 1로 설정
config.storage.topic=connect-configs
config.storage.replication.factor=1

# status를 저장할 Topic 설정. 
# this topic can have multiple partitions and should be replicated and compacted. 
# 해당 토픽은 자동으로 생성됨 
# default replication factor = 3 이며 특별한 케이스에서는 더 커질수도 있다
# 개발환경에서는 single-broker 로 움직이기 때문에 1로 설정
status.storage.topic=connect-status
status.storage.replication.factor=1
#status.storage.partitions=5

# 테스트/디버깅을 위해서 더 빠르게 설정해 놓습니다. 
offset.flush.interval.ms=1000
plugin.path=/usr/local/kafka/plugins
{% endhighlight %}


Distributed Mode 로 Connector를 실행시킵니다. 

{% highlight bash %}
# production 에서 daemon 으로 돌릴려면.. -daemon 을 맨 앞에 넣어서 돌리면 됨
# connect-distributed.sh -daemon connect-distributed.properties
$ connect-distributed.sh connect-distributed.properties


# 정상적으로 작동하는지 확인합니다.
$ curl -s localhost:8083 | jq
{
  "version": "2.8.1",
  "commit": "839b886f9b732b15",
  "kafka_cluster_id": "zrADZwVuRQ2CnAMIW_6DZw"
}


$ curl -s localhost:8083/connector-plugins | jq
[
  {
    "class": "org.apache.kafka.connect.file.FileStreamSinkConnector",
    "type": "sink",
    "version": "2.8.1"
  },
  {
    "class": "org.apache.kafka.connect.file.FileStreamSourceConnector",
    "type": "source",
    "version": "2.8.1"
  }
]

{% endhighlight %}


**connect-file-source.json**

{% highlight json %}
{
    "name": "local-file-source",
    "config": {
        "connector.class": "org.apache.kafka.connect.file.FileStreamSourceConnector",
        "tasks.max": 1,
        "file": "/home/anderson/Downloads/input.txt",
        "topic": "test-topic"
    }
}
{% endhighlight %}

{% highlight bash %}
$ curl  -d @"connect-file-source.json" \
        --header 'Content-Type: application/json' \
        --request POST 'localhost:8083/connectors' | jq

# 등록 확인
$ curl -s localhost:8083/connectors/local-file-source | jq
{% endhighlight %}


**connect-file-sink.json**

{% highlight json %}
{
    "name": "local-file-sink",
    "config": {
        "connector.class": "org.apache.kafka.connect.file.FileStreamSinkConnector",
        "tasks.max": 1,
        "file": "/home/anderson/Downloads/output.txt",
        "topics": "test-topic"
    }
}
{% endhighlight %}

{% highlight bash %}
$ curl  -d @"connect-file-sink.json" \
        --header 'Content-Type: application/json' \
        --request POST 'localhost:8083/connectors' | jq

# 등록 확인 
$ curl -s localhost:8083/connectors/local-file-sink | jq
{% endhighlight %}


**등록을 확인**합니다.

{% highlight bash %}
$ curl -s localhost:8083/connectors | jq
[
  "local-file-source",
  "local-file-sink"
]

# 로그 확인
$ docker logs -f kafka
{% endhighlight %}


**커넥터 삭제**는 다음과 같이 합니다.

{% highlight bash %}
$ curl -X DELETE http://localhost:8083/connectors/file-source-connector
{% endhighlight %}

확인은 다음과 같이 합니다.<br>
먼저 터미널을 새로 열고..

{% highlight bash %}
$ touch output.txt
$ tail -f output.txt
{% endhighlight %}

새로운 터미널을 열고..

{% highlight bash %}
$ touch input.txt
$ echo hello >> input.txt
{% endhighlight %}