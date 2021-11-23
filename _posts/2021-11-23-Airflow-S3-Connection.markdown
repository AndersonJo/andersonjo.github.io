---
layout: post 
title:  "Airflow S3 Connection"
date:   2021-11-23 01:00:00 
categories: "airflow"
asset_path: /assets/images/ 
tags: ['assumerole', 'role', 'iam', 'aws', 's3', 'sts']
---


# 1. Assume Role

Assuming a role 의미는 Security Token Service (STS)를 통해서 temporary credentials 을 제공받는 것 입니다. <br>

<img src="{{ page.asset_path }}role_01.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


생성된 이후 Edit Trust Relation 을 눌러서 수정합니다.

{% highlight json %}
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:user/anderson"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "ml-service"
        }
      }
    }
  ]
}
{% endhighlight %}

중요하게 볼 부분은 "AWS": "arn:aws:iam::123456789012:user/anderson" 부분이 있는데, <br>
가져올 유저의 ARN을 넣으면 됩니다. 


# 2. Airflow 

## 2.1 Configure Connection

Admin -> Connections 에서 S3 Connection 을 추가합니다.

<img src="{{ page.asset_path }}role_02.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

role_arn에는 새로 생성한 Role의 ARN을 넣으면 됩니다. 

{% highlight json %}
{"region_name": "ap-northeast-2",
 "role_arn": "arn:aws:iam::123456789012:role/ml-service",
 "assume_role_kwargs": {"ExternalId": "ml-service"}}
{% endhighlight %}


## 2.2 DAG Example

아래 코드는 S3 object (파일)이 올라오는지를 체크하는 코드입니다. <br>
S3 파일이 올라올때까지 기다린다고 생각하면 됩니다.<br>
여기서는 일단 S3 접속이 잘 되는지를 확인하면 됩니다.

{% highlight python %}
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.sensors.s3_key import S3KeySensor

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 11, 1),
    'email': ['anderson.pytorch@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

with DAG('s3_dag_test', default_args=default_args, schedule_interval='@once') as dag:
    t1 = BashOperator(
        task_id='bash_test',
        bash_command='echo "hello, it should work" > s3_conn_test.txt')

    sensor = S3KeySensor(
        task_id='check_s3_for_file_in_s3',
        bucket_key='s3://ml-data/2021082804432382.jpg',
        wildcard_match=True,
        aws_conn_id='aws_s3',
        timeout=60*60,
        poke_interval=10)

    t1 >> sensor
{% endhighlight %}

실행하면 다음과 같은 결과가 나와야 합니다. 

{% highlight bash %}
$ airflow dags test s3_dag_test now
INFO - ... | finished run 1 of 1 | tasks waiting: 0 | succeeded: 2 | running: 0 | failed: 0 | skipped: 0 | deadlocked: 0 | not ready: 0
{% endhighlight %}