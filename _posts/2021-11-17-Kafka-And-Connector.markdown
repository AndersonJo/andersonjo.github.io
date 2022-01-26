---
layout: post 
title:  "Apache Kafka Using Docker & Connector"
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
{% endhighlight %}

정상 작동 확인합니다.

{% highlight bash %}
# 정상적으로 작동하는지 확인합니다.
$ curl -s localhost:8083 | jq
{
  "version": "2.8.1",
  "commit": "839b886f9b732b15",
  "kafka_cluster_id": "zrADZwVuRQ2CnAMIW_6DZw"
}

# 사용 가능한 plugins 리스트 
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

















## 2.3 ESK Distributed Mode



- NOT_ENOUGH_REPLICAS 에러가 발생하면 connect-offsets, connect-configs, 그리고 connect-status <br>
  이 3개의 topics의 replication.factor 의 값이 min.insync.replica 보다 작아서 발생하는 문제입니다.<br>
  이 문제가 발생할시 3개의 토픽을 삭제하고 replication.factor=3 으로 맞추고 다시 생성하면 됩니다 .
- 2.2의 distributed mode 와 모두 동일하지만 replication 설정만 조금 다릅니다. 



먼저 min.insync.replicas 를 설정한 Topic을 생성합니다.

{% highlight bash %}
$ kafka-topics.sh --zookeeper $ZooKeeperConnect --create --topic test-topic \
                  --config min.insync.replicas=1 --partitions 10 --replication-factor 3
{% endhighlight %}

**connect-distributed.properties**

{% highlight yaml %}
bootstrap.servers=<Bootstrap Connect>

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
offset.storage.replication.factor=3
#offset.storage.partitions=25

# connector 그리고 task configuration을 저장할 Topic을 설정합니다. 
# this should be a single partition, highly replicated and compacted topic. 
# 해당 토픽은 자동으로 생성됨
# default replication factor = 3 이며 특별한 케이스에서는 더 커질수도 있다
# 개발환경에서는 single-broker 로 움직이기 때문에 1로 설정
config.storage.topic=connect-configs
config.storage.replication.factor=3

# status를 저장할 Topic 설정. 
# this topic can have multiple partitions and should be replicated and compacted. 
# 해당 토픽은 자동으로 생성됨 
# default replication factor = 3 이며 특별한 케이스에서는 더 커질수도 있다
# 개발환경에서는 single-broker 로 움직이기 때문에 1로 설정
status.storage.topic=connect-status
status.storage.replication.factor=3
#status.storage.partitions=5

# 테스트/디버깅을 위해서 더 빠르게 설정해 놓습니다.
offset.flush.interval.ms=1000
plugin.path=/usr/local/kafka/plugins
{% endhighlight %}











# 3. Docker Kafka and Connector Tutorial

## 3.1 Setting environment variables 

아래 Zookeeper, Kafka, Schema Registry, Rest Proxy, Connect 등등  properties 내용을 환경변수로 대체 사용 가능 합니다.
다만 아래처럼 글자 치환이 필요 합니다.

- kafka.properties를 환경변수로 대체해서 사용 
  - prefix: `KAFKA_`
  - replace: `.` -> `_`
  - replace: `--` -> `__` (2 underscores)
  - replace: `_` -> `___` (3 underscores)
  

## 3.2 Zookeeper 

MSK사용시 실행할 필요 없습니다.

- ZOOKEEPER_CLIENT_PORT: (required)
- ZOOKEEPER_SERVER_ID: cluster mode 사용시 필요.
- 자세한 내용은 [링크](https://docs.confluent.io/platform/current/installation/docker/config-reference.html) 참조

{% highlight bash %}
$ docker run -d \
--net=host \
--name=zookeeper \
-e ZOOKEEPER_CLIENT_PORT=32181 \
-e ZOOKEEPER_TICK_TIME=2000 \
-e ZOOKEEPER_SYNC_LIMIT=2 \
confluentinc/cp-zookeeper:7.0.1
{% endhighlight %}

## 3.3 Kafka

{% highlight bash %}
$ docker run -d \
    --net=host \
    --name=kafka \
    -e KAFKA_ZOOKEEPER_CONNECT=localhost:32181 \
    -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:29092 \
    -e KAFKA_BROKER_ID=2 \
    -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
    confluentinc/cp-kafka:7.0.1
{% endhighlight %}

## 3.4 Schema Registry

{% highlight bash %}
$ docker run -d \
  --net=host \
  --name=schema-registry \
  -e SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS=SSL://hostname2:9092 \
  -e SCHEMA_REGISTRY_HOST_NAME=localhost \
  -e SCHEMA_REGISTRY_LISTENERS=http://localhost:8081 \
  -e SCHEMA_REGISTRY_DEBUG=true \
  confluentinc/cp-schema-registry:7.0.1
{% endhighlight %}


## 3.5 Rest Proxy

{% highlight bash %}
$ docker run -d \
  --net=host \
  --name=kafka-rest \
  -e KAFKA_REST_ZOOKEEPER_CONNECT=localhost:32181 \
  -e KAFKA_REST_LISTENERS=http://localhost:8082 \
  -e KAFKA_REST_SCHEMA_REGISTRY_URL=http://localhost:8081 \
  -e KAFKA_REST_BOOTSTRAP_SERVERS=localhost:29092 \
  confluentinc/cp-kafka-rest:7.0.1
{% endhighlight %}


## 3.6 Create Topics

Kafka Connect는 config, status, offsets of the connectors 등의 정보들을 모두 Kafka Topics 에 저장해 둡니다.

{% highlight bash %}
$ kafka-topics.sh --create \
        --topic quickstart-avro-offsets \
        --partitions 1 \
        --replication-factor 1 \
        --config cleanup.policy=compact \
        --if-not-exists --bootstrap-server localhost:29092

$ kafka-topics.sh --create \
        --topic quickstart-avro-config \
        --partitions 1 \
        --replication-factor 1 \
        --config cleanup.policy=compact \
        --if-not-exists --bootstrap-server localhost:29092

$ kafka-topics.sh --create \
        --topic quickstart-avro-status \
        --partitions 1 \
        --replication-factor 1 \
        --config cleanup.policy=compact \
        --if-not-exists --bootstrap-server localhost:29092

$ kafka-topics.sh --describe --bootstrap-server localhost:29092
{% endhighlight %}


## 3.7 Connect 

먼저 운영에 필요한 jars 파일들을 다운받아서 Docker에 넣어야 합니다.

{% highlight bash %}
# MySQL Connectors
$ mkdir -p /tmp/quickstart/jars
$ curl -k -SL "http://dev.mysql.com/get/Downloads/Connector-J/mysql-connector-java-5.1.37.tar.gz" | tar -xzf - -C /tmp/quickstart/jars --strip-components=1 mysql-connector-java-5.1.37/mysql-connector-java-5.1.37-bin.jar
$ curl -k -SL "https://downloads.mysql.com/archives/get/p/3/file/mysql-connector-java-5.1.49.tar.gz" | tar -xzf - -C /tmp/quickstart/jars --strip-components=1 mysql-connector-java-5.1.49/mysql-connector-java-5.1.49-bin.jar
$ curl -k -SL "https://downloads.mysql.com/archives/get/p/3/file/mysql-connector-java-8.0.27.tar.gz" | tar -xzf - -C /tmp/quickstart/jars --strip-components=1 mysql-connector-java-8.0.27/mysql-connector-java-8.0.27.jar
{% endhighlight %}


{% highlight bash %}
$ docker run -d \
  --name=kafka-connect-avro \
  --net=host \
  -e CONNECT_BOOTSTRAP_SERVERS=localhost:29092 \
  -e CONNECT_REST_PORT=28083 \
  -e CONNECT_GROUP_ID="quickstart-avro" \
  -e CONNECT_CONFIG_STORAGE_TOPIC="quickstart-avro-config" \
  -e CONNECT_OFFSET_STORAGE_TOPIC="quickstart-avro-offsets" \
  -e CONNECT_STATUS_STORAGE_TOPIC="quickstart-avro-status" \
  -e CONNECT_CONFIG_STORAGE_REPLICATION_FACTOR=1 \
  -e CONNECT_OFFSET_STORAGE_REPLICATION_FACTOR=1 \
  -e CONNECT_STATUS_STORAGE_REPLICATION_FACTOR=1 \
  -e CONNECT_KEY_CONVERTER="io.confluent.connect.avro.AvroConverter" \
  -e CONNECT_VALUE_CONVERTER="io.confluent.connect.avro.AvroConverter" \
  -e CONNECT_KEY_CONVERTER_SCHEMA_REGISTRY_URL="http://localhost:8081" \
  -e CONNECT_VALUE_CONVERTER_SCHEMA_REGISTRY_URL="http://localhost:8081" \
  -e CONNECT_INTERNAL_KEY_CONVERTER="org.apache.kafka.connect.json.JsonConverter" \
  -e CONNECT_INTERNAL_VALUE_CONVERTER="org.apache.kafka.connect.json.JsonConverter" \
  -e CONNECT_REST_ADVERTISED_HOST_NAME="localhost" \
  -e CONNECT_LOG4J_ROOT_LOGLEVEL=DEBUG \
  -e CONNECT_PLUGIN_PATH=/usr/share/java,/etc/kafka-connect/jars \
  -v /tmp/quickstart/file:/tmp/quickstart \
  -v /tmp/quickstart/jars:/etc/kafka-connect/jars \
  confluentinc/cp-kafka-connect:latest
{% endhighlight %}

정상 작동했는지 확인은 다음과 같이 합니다. (대충 요렇게 나옵니다)

{% highlight bash %}
$ docker logs kafka-connect-avro  | egrep   "(Connect|Herder) started"
[2022-01-25 09:26:15,948] INFO Kafka Connect started 
[2022-01-25 09:26:16,780] INFO [Worker clientId=connect-1, groupId=quickstart-avro] Herder started 
{% endhighlight %}



## 3.8 MySQL 

Database를 실행시킵니다.

{% highlight bash %}
$ docker run -d \
  --name=quickstart-mysql \
  --net=host \
  -e MYSQL_ROOT_PASSWORD=confluent \
  -e MYSQL_USER=confluent \
  -e MYSQL_PASSWORD=confluent \
  -e MYSQL_DATABASE=connect_test \
  mysql

$ docker exec -it quickstart-mysql bash
$ mysql -u confluent -pconfluent
{% endhighlight %}


{% highlight sql %}
CREATE DATABASE IF NOT EXISTS connect_test;
USE connect_test;

DROP TABLE IF EXISTS test;


CREATE TABLE IF NOT EXISTS test (
  id serial NOT NULL PRIMARY KEY,
  name varchar(100),
  email varchar(200),
  department varchar(200),
  modified timestamp default CURRENT_TIMESTAMP NOT NULL,
  INDEX `modified_index` (`modified`)
);

INSERT INTO test (name, email, department) VALUES ('alice', 'alice@abc.com', 'engineering');
INSERT INTO test (name, email, department) VALUES ('bob', 'bob@abc.com', 'sales');
INSERT INTO test (name, email, department) VALUES ('bob', 'bob@abc.com', 'sales');
INSERT INTO test (name, email, department) VALUES ('bob', 'bob@abc.com', 'sales');
INSERT INTO test (name, email, department) VALUES ('bob', 'bob@abc.com', 'sales');
INSERT INTO test (name, email, department) VALUES ('bob', 'bob@abc.com', 'sales');
INSERT INTO test (name, email, department) VALUES ('bob', 'bob@abc.com', 'sales');
INSERT INTO test (name, email, department) VALUES ('bob', 'bob@abc.com', 'sales');
INSERT INTO test (name, email, department) VALUES ('bob', 'bob@abc.com', 'sales');
INSERT INTO test (name, email, department) VALUES ('bob', 'bob@abc.com', 'sales');
exit;
{% endhighlight %}

## 3.9 Source

{% highlight bash %}
$ export CONNECT_HOST=localhost
$ curl -X POST \
  -H "Content-Type: application/json" \
  --data '{ "name": "quickstart-jdbc-source", "config": { "connector.class": "io.confluent.connect.jdbc.JdbcSourceConnector", "tasks.max": 1, "connection.url": "jdbc:mysql://127.0.0.1:3306/connect_test?user=root&password=confluent", "mode": "incrementing", "incrementing.column.name": "id", "timestamp.column.name": "modified", "topic.prefix": "quickstart-jdbc-", "poll.interval.ms": 1000 } }' \
  http://$CONNECT_HOST:28083/connectors
{% endhighlight %}