---
layout: post 
title:  "Airflow Connections"
date:   2021-11-23 01:00:00 
categories: "airflow"
asset_path: /assets/images/ 
tags: ['assumerole', 'role', 'iam', 'aws', 's3', 'sts']
---


# 1. Connections

## 1.1 MySQL (AWS RDS)

**Airflow Connection**

- Connection ID: mysql_conn_test
- Connection Type: MySQL
- Host: Host 주소
- Schema: Database 이름
- Login: User 이름
- Password: 패스워드
- Port: 3306
- Extra: `{"charset": "utf8mb4", "database_type": "mysql", "use_proxy": false}`

Extra 부분에서 charset이 중요합니다. <br> 
이유는 일단 airflow 는 SQL을 encoding시에 latin1으로 하는듯 합니다. (즉 기본값이 utf-8 아닙니다.)<br>
utf8mb4 가 안되면 utf-8 도 해보면 됩니다. 


**Connection Test**

{% highlight python %}
from airflow.providers.mysql.hooks.mysql import MySqlHook

hook = MySqlHook(mysql_conn_id='mysql_conn_test')
assert hook.test_connection()
{% endhighlight %}

**Pandas 데이터 받기**

{% highlight python %}
from airflow.providers.mysql.hooks.mysql import MySqlHook

SQL = 'SELECT * FROM test LIMIT 10'
hook = MySqlHook(mysql_conn_id='mysql_conn_test')
df = hook.get_pandas_df(SQL)
{% endhighlight %}

**Data Transfer from RDBMS to S3**

{% highlight python %}
from airflow.providers.amazon.aws.transfers.mysql_to_s3 import MySQLToS3Operator

SQL = 'SELECT * FROM test LIMIT 10'
t2 = MySQLToS3Operator(
    task_id='rdbms_to_s3',
    query=SQL,
    s3_bucket='bucket_name',
    s3_key='data.parquet',
    mysql_conn_id='mysql_conn_test',
    aws_conn_id='aws_s3',
    file_format='parquet')
{% endhighlight %}

 - s3_bucket: 그냥 이름 넣으면 됨. (ARN 이나 s3:// 이런거 아님)
 - s3_key: 파일이름 (.parquet 또는 .csv)
 - file_format: csv 또는 parquet

## 1.2 GCP (BigQuery)

**GCP Service Account**

1. IAM -> Service Accounts -> Create Service Account 
2. 생성중 `Grant this service account access to project` 부분이 매우매우 중요
   1. `BigQuery Job User` 반드시 추가
   2. `BigQuery Data Editor` <- BigQuery 테이블 생성시 필요
3. 생성 완료후 -> 새로 생성된 service account에서 manage keys 를 눌러서 -> 새로운 키 생성해서 json 다운로드 받음

만약 생성이후 권한 추가시에는 IAM -> IAM 에서 권한을 추가해야지, Service Accounts에서 Principal 에서 권한 추가하면 안됨. 


**Airflow Connection**

 - Connection Id: my-gcp
 - Connection Type: `Google Cloud` (Google BigQuery 도 있는데 이거 아님)
 - Keyfile Path: `/home/anderson/gcp-f123456789.json` 같은 private key의 파일 위치를 넣으면 됨
 - Keyfile JSON: Keyfile Path안에 있는 json 파일 내용 전체를 카피해서 넣으면 됨 (Keyfile Path보다 이게 좋음 - 파일 관리 필요없음)
 - Number of Retries: 5
 - Project Id: GCP의 project ID를 넣으면 됨 (예. zeta-1234556)
 - Scopes: (Optional이고 없어도 됨) `https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/bigquery`

**BigQuery Connection Test**

{% highlight python %}
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook

bigquery = BigQueryHook(gcp_conn_id='my-gcp')
print(bigquery.test_connection())
# (True, 'Connection successfully tested')
{% endhighlight %}


**BigQuery Create Table**

{% highlight python %}
from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateEmptyTableOperator

t1 = BigQueryCreateEmptyTableOperator(
        task_id='bigquery_create_table',
        bigquery_conn_id='my-gcp',
        project_id='gcp-project-id',
        dataset_id='test_dataset',
        table_id='product_score',
        schema_fields=[{'name': 'name', 'type': 'STRING', 'mode': 'REQUIRED'},
                       {'name': 'value', 'type': 'INTEGER', 'mode': 'NULLABLE'}]
    )
{% endhighlight %}

<img src="{{ page.asset_path }}airflow_connection_01.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

추가적으로..
 - time_partitioning: 파티셔닝
 - gcs_schema_object: `gs://test-bucket/dir1/dir2/employee_schema.json` 처럼 GCS에서 가져오게 할 수도 있다

# 2. AWS

## 2.1 Assume Role

Assuming a role 의미는 Security Token Service (STS)를 통해서 temporary credentials 을 제공받는 것 입니다. <br>

<img src="{{ page.asset_path }}role_01.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


생성된 이후 Edit Trust Relation 을 눌러서 수정합니다.

{% highlight json %} {
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
} } }
]
} {% endhighlight %}

중요하게 볼 부분은 "AWS": "arn:aws:iam::123456789012:user/anderson" 부분이 있는데, <br>
가져올 유저의 ARN을 넣으면 됩니다.

## 2.2 Configure Connection

Admin -> Connections 에서 S3 Connection 을 추가합니다.

<img src="{{ page.asset_path }}role_02.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

role_arn에는 새로 생성한 Role의 ARN을 넣으면 됩니다.

{% highlight json %} {"region_name": "ap-northeast-2",
"role_arn": "arn:aws:iam::123456789012:role/ml-service",
"assume_role_kwargs": {"ExternalId": "ml-service"}} {% endhighlight %}

아래 코드는 S3 object (파일)이 올라오는지를 체크하는 코드입니다. <br>
S3 파일이 올라올때까지 기다린다고 생각하면 됩니다.<br>
여기서는 일단 S3 접속이 잘 되는지를 확인하면 됩니다.

{% highlight python %} from airflow import DAG from datetime import datetime, timedelta from airflow.operators.bash
import BashOperator from airflow.providers.amazon.aws.sensors.s3_key import S3KeySensor

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
task_id='bash_test', bash_command='echo "hello, it should work" > s3_conn_test.txt')

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

