---
layout: post 
title:  "AWS MSK (Managed Streaming for Apache Kafka)"
date:   2021-11-15 01:00:00 
categories: "data-engineering"
asset_path: /assets/images/ 
tags: []
---

<header>
    <img src="{{ page.asset_path }}kafka_background.jpeg" class="center img-responsive img-rounded img-fluid">
</header>

# 1. Architecture

<img src="{{ page.asset_path }}msk-architecture.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

- **Broker Nodes**
  - MSK 생성시 각각의 Availability Zone 안에 몇개의 Broker Node를 생성할지 설정 할 수 있다. 
  - 위의 그림에서는 각각의 availability zone마다 1개의 broker node가 존재하고 있으며,<br>각각의 availability zone은 각각의 VPC subnet을 갖고 있다
- **ZooKeeper Nodes**
  - Highly reliable distributed coordination 을 구현
- **Producers, Consumers and Topic Creators**
  - Topic 생성 및 데이터 produce 또는 consume 을 하는데 사용
- **Cluster Operations**
  - CLI 또는 SDK를 통해서 control-plane operations 을 수행할수 있습니다. 
  - 예를 들어 cluster 생성, 삭제 또는 클러스터 리스팅, 클러스터 속성 보기, 또는 Broker 변경등을 수행할 수 있습니다. 

# 2. VPC Configureation

## 2.1 Create a VPC for MSK Cluster

[https://console.aws.amazon.com/vpc/](https://console.aws.amazon.com/vpc/) 링크로 들어가서 Launch VPC Wizrd 를 선택합니다.<br>

먼저 **첫번째 Subnet**을 생성합니다.
1. 설정
   1. **VPC name**: `kafka_vpc_test`
   2. **Availability Zone**: `us-east-2a` 선택
   3. **Subnet name**: `kafka_vpc_subnet_01`
   4. Create VPC 선택

<img src="{{ page.asset_path }}kafka_01.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">





Kafka VPC ID(`vpc-123456789abcdefgh`) 는 따로 적어놓습니다. <br>

<img src="{{ page.asset_path }}kafka_02.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


## 2.2 High Availability and Fault Tolerance 

High availability 그리고 Fault Tolerance를 구현하기 위해서 추가적으로 다른 availability zones 에 2개의 subnet을 추가합니다.



**두번째 Subnet**을 생성합니다.<br>
두번째 subnet은 다음과 같이 생성합니다. 

1. Subnets 메뉴를 선택하고 이전에 생성한 `kafka_vpc_subnet_01` 을 찾고 -> Route Table을 찾고 복사 (`rtb-0fabcd123456789ff`)합니다. 
2. Create Subnet 버튼을 누릅니다.
   1. VPC ID: 이전에 생성한 VPC ID (`kafka_vpc_test`) 를 선택
   2. **Subnet name**: `kafka_vpc_subnet_02`
   3. **Availability Zone**: `us-east-2b`
   4. **CIDR block**: `10.0.1.0/24` 
   5. 생성! 

<img src="{{ page.asset_path }}kafka_03.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


바로 이전에 만든 subnet을 선택하고, 아래 메뉴에서 `Route table`을 선택후 `Edit route table association`을 선택 

<img src="{{ page.asset_path }}kafka_04.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

Route Table ID 드랍박스에서, 첫번째 만들었던 route table id(`rtb-0fabcd123456789ff`)을 선택하고 저장합니다.


<img src="{{ page.asset_path }}kafka_05.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

동일한 방식으로 Availability Zone 만 바꿔서 생성합니다. 
   1. VPC ID: 이전에 생성한 VPC ID (`kafka_vpc_test`) 를 선택
   2. **Subnet name**: `kafka_vpc_subnet_03`
   3. **Availability Zone**: `us-east-2c`
   4. **CIDR block**: `10.0.2.0/24` 


## 2.3 Create Cluster

MSK 서비스로 이동후 Create Cluster 를 누릅니다.

1. Custom Create 선택
2. **Cluster name**: `kafka-test`
3. **Apache Kafka Version**: 2.8.1
4. **Configuration**: MSK default configuration
5. **Networking**: 이전의 생성한 Subnets을 선택합니다.


<img src="{{ page.asset_path }}kafka_10.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

<img src="{{ page.asset_path }}kafka_11.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

<img src="{{ page.asset_path }}kafka_12.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

<img src="{{ page.asset_path }}kafka_13.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">




# 3. Create Topic

## 3.1 Create EC2 

AWS MSK는 좀 특이하게도.. 외부 접속이 되지를 않습니다.<br> 
내부적으로, MSK는 VPC zone 내부에 존재하고 있으며 Zookeep, broker url은 private ip로 사용됩니다. <br> 
VPC 내부의 EC2를 생성해서 접속해야 합니다.

1. Configure Instance
   1. **Network**: 이전에 설정한 kafka_vpc_test 를 선택합니다. 
   2. **Auto-assign Public IP**: Enable 
   
<img src="{{ page.asset_path }}kafka_14.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


그 다음에 할일은 Kafka Cluster 의 Security Group 의 inbound 를 편집합니다. <br>
여기서 All Traffic 으로 하되 Source를 새로 생성한 EC2 instance의 security group 을 넣으면 됩니다.<br>
자세한 방법은 아래와 같습니다. 


1. EC2 Instance 
   1. Instances 에서 생성된 instance를 클릭하고 Security 메뉴에서 Security Groups 를 어딘가에 복사합니다.
   2. `sg-instance-security-group` 으로 이름을 예제로 사용하겠습니다. 

2. Kafka Cluster
   1. Cluster 선택
   2. Networking -> Security groups applied 를 찾습니다. 
   3. 예제에서는 `sg-kafka-cluster-security-group` 으로 사용하겠습니다. 
   4. 그냥 클릭합니다. 

3. Kafka Cluster 의 Security Group의 INBOUND 편집
   1. Security Group  
   2. Inbound Rules 탭 
   3. Edit Inbound Rules 선택
   4. Add Rule 선택
      1. Type: All traffic
      2. Source: Custom 선택후 복사한 Security Group을 넣습니다. 



## 3.1 Installation

.bashrc 에 다음과 같이 `JAVA_HOME`을 설정해 줍니다. 

{% highlight bash %}
$ sudo apt-get install openjdk-11-jdk
$ sudo apt install mlocate
{% endhighlight %}

추가적으로 aws cli 그리고 aws-mfa 설치해야 합니다. <br>
다음의 내용을 ~/.bashrc 에 넣습니다. 

{% highlight bash %}
export PATH=$PATH:/home/ubuntu/.local/bin/
export JAVA_HOME=/usr/lib/jvm/java-1.11.0-openjdk-amd64
{% endhighlight %}



{% highlight bash %}
# EC2 Instance
# 2.8.1 인 경우
$ wget https://archive.apache.org/dist/kafka/2.8.1/kafka_2.12-2.8.1.tgz

# 2.6.2 인 경우
$ wget https://archive.apache.org/dist/kafka/2.6.2/kafka_2.12-2.6.2.tgz

$ tar -xzf kafka_2.12-2.8.1.tgz
$ cd kafka_2.12-2.8.1
{% endhighlight %}


다시 로컬 환경으로 와서 ZookeeperConnectString 값을 확인합니다. <br>
CLUSTER ARN을 복사하고 다음의 명령어로 클러스터를 확인합니다.<br>
CLUSTER_ARN은 변경해야 합니다.

{% highlight bash %}
# Local Computer
$ aws kafka describe-cluster --region us-east-2 --cluster-arn CLUSTER_ARN 
{% endhighlight %}

## 3.2 Create Topic

다음의 명령어로 Zookeeper Connect 를 알아냅니다. 

{% highlight bash %}
# Local Computer
$ aws kafka describe-cluster --region us-east-2 --cluster-arn CLUSTER ARN | grep ZookeeperConnectString
{% endhighlight %}

다음과 같이 생성합니다. <br>
ZookeeperConnectString 부분은 위에서 grep으로 잡은 전체 정보를 넣어야 합니다. 

{% highlight bash %}
# EC2 Instance 
$ bin/kafka-topics.sh --create --zookeeper ZookeeperConnectString --replication-factor 3 --partitions 1 --topic TestTopic
Created topic TestTopic.
{% endhighlight %}


## 3.4 Broker URL

아래에서 ClusterArn 는 수정해야 합니다.

{% highlight bash %}
# Local 
$ aws kafka get-bootstrap-brokers --region us-east-2 --cluster-arn ClusterArn
{
    "BootstrapBrokerString": "b-2.kafka-test.allwn4.c3.kafka.us-east-2.amazonaws.com:9092,b-1.kafka-test.allwn4.c3.kafka.us-east-2.amazonaws.com:9092,b-3.kafka-test.allwn4.c3.kafka.us-east-2.amazonaws.com:9092",
    "BootstrapBrokerStringTls": "b-2.kafka-test.allwn4.c3.kafka.us-east-2.amazonaws.com:9094,b-1.kafka-test.allwn4.c3.kafka.us-east-2.amazonaws.com:9094,b-3.kafka-test.allwn4.c3.kafka.us-east-2.amazonaws.com:9094",
    "BootstrapBrokerStringSaslIam": "b-2.kafka-test.allwn4.c3.kafka.us-east-2.amazonaws.com:9098,b-1.kafka-test.allwn4.c3.kafka.us-east-2.amazonaws.com:9098,b-3.kafka-test.allwn4.c3.kafka.us-east-2.amazonaws.com:9098"
}

{% endhighlight %}

## 3.3 SSH Tunnelling 

MSK의 가장 큰 문제점은 역시.. 외부에서 topic 연결이 안된다는 것이고.<br> 
이를 해결하는 방법은 많은데.. 제가 사용하는 방법은 그냥 ssh tunneling을 사용하는 것 입니다.

기본적으로 `ssh -i aws.pem -N -L {local port}:{MSK Broker}:9092 ubuntu@{EC2 address}` 이렇게 사용합니다.<br>
예제에서는 3개의 broker를 사용하기 때문에.. "-L {local port}:{MSK Broker}:9092" 방식으로 추가를 더 해주면 됩니다. 

{% highlight bash %}
ssh -i ~/.ssh/aws.pem -N -L 9093:b-3.kafka-test.allwn4.c3.kafka.us-east-2.amazonaws.com:9092 ubuntu@ec2-5-20-100-100.us-east-2.compute.amazonaws.com
{% endhighlight %}

> 근데 실제 사용해보니.. SSH Tunneling은 매우 느립니다.<br>
> 가장 쉽게 해결할수 있는 방법은 EC2에 OpenVPN을 설치해서 연결하는게 가장 편리합니다. 

## 3.4 Connection from outside 

SSH Tunneling은 지역에 따라 느릴수 있습니다.<br>
[링크](https://awsfeed.com/whats-new/big-data/secure-connectivity-patterns-to-access-amazon-msk-across-aws-regions) 에서 다른 방법으로 접속하는 방법을 잘 설명하고 있습니다.<br>




## 3.3 Python Example 

{% highlight bash %}
pip3 install kafka-python
{% endhighlight %}


producer.py 는 다음과 같이 작성합니다.

{% highlight python %}
import json
from kafka import KafkaProducer

def produce():
producer = KafkaProducer(
        acks=0,
        bootstrap_servers=[
            'localhost:9091'
            'localhost:9092',
            'localhost:9093'],
        api_version=(2, 8, 1),
        value_serializer=lambda x: json.dumps(x).encode('utf-8'))

    for i in range(10):
        print(f'Sending: {i}')
        producer.send('TestTopic', value=f'test {i}')
    producer.flush()

if __name__ == '__main__':
    produce()
{% endhighlight %}