---
layout: post
title:  "Amazon Kinesis Data Streams"
date:   2020-10-10 01:00:00
categories: "aws"
asset_path: /assets/images/
tags: ['kafka', 'spark', 'realtime']
---

# 1. Introduction

## 1.1 Architecture 

<img src="{{ page.asset_path }}kinesis-architecture.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

 - Kinesis는 Shards로 이루어져 있음  
 - **Data Record**: Kinesis안에서 일종의 데이터 유닛이며, [sequence number](https://docs.aws.amazon.com/streams/latest/dev/key-concepts.html#sequence-number) 그리고 [Partition Key](https://docs.aws.amazon.com/streams/latest/dev/key-concepts.html#partition-key) 으로 구성
 - **Retention Period**: 기본값 24시간이고, 7일까지 늘릴수 있음
 - **Producer & Consumer**: 보내고 받는 받는 쪽이고, Consumer의 경우 [Stream Application](https://docs.aws.amazon.com/streams/latest/dev/key-concepts.html#enabled-application)이라고도 함 
 - **Shard**
     - Read: 초당 5 transactions, 그리고 2MB 데이터를 읽을 수 있음
     - Write: 초당 1000 records 를 1MB 까지 (partition key 포함) write 할 수 있음 
     - 만약 처리하는 데이터가 증가한다면 [Resharding](https://docs.aws.amazon.com/streams/latest/dev/kinesis-using-sdk-java-resharding.html)이 필요
 - **Put Record**: (Data, StreamName, PartitionKey) <- Producer는 PUT call을 함으로서 데이터를 올림
 - **Partition Key**: producer에 의해서 제공되며, 여러 Shards에 분배시키기 위해서 사용됨
 
## 1.2 Two Types of Consumers 

두가지 종류의 Consumers가 존재합니다. 

1. 그냥 fan-out consumers
    - **Shard Read Throughput**: Shard당 2 MB/sec 고정되어 있으며, 다수의 consumers가 동일한 shard에서 가져갈시(read) 모두 동일한 데이터를 가져가지만, 초당 2MB 을 넘어가면 안됨
    - **Message Propagation Delay**: 하나의 consumer당 200ms 가 걸리고, 5개의 consumers가 존재시 1000ms 까지 느려질수 있음
    - **Cost**: 없음 
2. enhanced fan-out consumers 
    - 기존 fan-out consumers의 제약을 넘어서기 위해서 나온 consumer 
    - **Shard Read Throughput**: enhanced fan-out consumers는 모두 **독립적인** read throughput을 갖으며 각각 2 MB/Sec 를 갖는다
    - **Message Propagation Delay**: 하나의 consumer가 있는 5개가 있든 상관없이 평균 70ms 가 걸림
    - **Cost**: 데이터를 retrival 그리고 consumer-shard hour cost 존재




# 2. Getting Started 

## 2.1 AWS Kinesis CLI 

**Put and Get a Record** 

{% highlight bash %}
# Stream 생성
$ aws kinesis create-stream --stream-name Test --shard-count 1

# Stream 정보 얻기
$ aws kinesis describe-stream-summary --stream-name Test 

# Put
$ aws kinesis put-record --stream-name Test  --partition-key 123 --data testdata

# ShardIterator 얻기 
$ aws kinesis get-shard-iterator --shard-id shardId-000000000000 --shard-iterator-type TRIM_HORIZON --stream-name Test
{
    "ShardIterator": "AAAAAAAAAAEStPmsRs6uXnop71REHTk4zDK+eIVTk2Vl7AI+w6+F6Y2K1UA6igTB6O6SR2ohZBo1YeL971sPIzvR3LQprRDsGd5uFSYJJDsDtUU9NmqPQhyotNToSkV6h4dGyAMH2JY8ULA8rt+lLp/xLPoaLGY7RkUsFBPDN6A6PMnpc01E0tr7zPc5Bam3C2pdm1Rrnu+PcZ9MLInkTURKMza5JYqQ"
}

# 또는 이렇게.. 
SHARD_ITERATOR=$(aws kinesis get-shard-iterator --shard-id shardId-000000000000 --shard-iterator-type TRIM_HORIZON --stream-name Test --query 'ShardIterator')

# Get (ShardIterator값을 복사에서 넣는다) 

$ aws kinesis get-records --shard-iterator AAAAAAAAAAEStPmsRs6uXnop71REHTk4zDK+eIVTk2Vl7AI+w6+F6Y2K1UA6igTB6O6SR2ohZBo1YeL971sPIzvR3LQprRDsGd5uFSYJJDsDtUU9NmqPQhyotNToSkV6h4dGyAMH2JY8ULA8rt+lLp/xLPoaLGY7RkUsFBPDN6A6PMnpc01E0tr7zPc5Bam3C2pdm1Rrnu+PcZ9MLInkTURKMza5JYqQ 

{% endhighlight %}



## 2.2 IAM Policy and User

 - [IAM Policy and User for Kinesis](https://docs.aws.amazon.com/streams/latest/dev/tutorial-stock-data-kplkcl2-iam.html)

아래의 Test Stream에서 ARN을 찾고 복사합니다.

<img src="{{ page.asset_path }}kinesis-test-arn.png" class="img-responsive img-rounded img-fluid " style="border: 2px solid #333333">

IAM -> Policy -> Create Policy 를 한 후  Service 는 Kinesis 선택하고, 필요한 권한들을 넣습니다. 

 - DescribeStream
 - GetShardIterator
 - GetRecords
 - PutRecord
 - PutRecords
 
Stream 에다가 TestStream ARN을 넣습니다. 

<img src="{{ page.asset_path }}kinesis-create-policy.png" class="img-responsive img-rounded img-fluid " style="border: 2px solid #333333">

Policy가 만들어지면 User에다가 넣어주면 됩니다.

## 2.3 Python Code Snippet 

아래는 Producer 그리고 Consumer의 Python 예제 입니다.

**Producer**

{% highlight python %}
import json
import random
from time import sleep

from boto import kinesis as boto_kinesis
from faker import Faker

def generate_data(faker):
    return {'name': faker.name(),
            'age': random.randint(10, 20),
            'gender': random.choice(['M', 'F']),
            'score': random.choice(range(40, 70, 5)),
            'job': faker.job()}

def main():
    faker = Faker()
    kinesis = boto_kinesis.connect_to_region('us-east-2')
    print('Connected')

    if 'AndersonStream' not in kinesis.list_streams()['StreamNames']:
        kinesis.create_stream('AndersonStream', 1)

    for _ in range(50):
        data = generate_data(faker)
        res = kinesis.put_record('AndersonStream', json.dumps(data), 'partitionkey')
        print('PUT', data)
{% endhighlight %}


**Consumer**

{% highlight python %}
from boto import kinesis as boto_kinesis
def main():
    kinesis = boto_kinesis.connect_to_region('us-east-2')

    shard_id = 'shardId-000000000000'  # Shard는 1개만 갖고 있음
    shard_it = kinesis.get_shard_iterator('AndersonStream', shard_id, 'LATEST')['ShardIterator']
    print('Latest Shard Iterator:', shard_it)

    while True:
        _out = kinesis.get_records(shard_it, limit=10)
        records = _out['Records']

        for r in records:
            print(r['Data'])

        shard_it = _out['NextShardIterator']
        if not records:
            break
{% endhighlight %}