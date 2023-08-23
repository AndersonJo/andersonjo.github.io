---
layout: post 
title:  "Airflow Docker Compose & CLI List"
date:   2021-11-24 01:00:00 
categories: "airflow"
asset_path: /assets/images/ 
tags: ['docker']
---

# Airflow in Docker Compose 

## Install docker-compose.yaml 

아래와 같이 다운로드 받고, 설정을 합니다. 

```bash
$ curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.6.3/docker-compose.yaml'
$ mkdir -p ./dags ./logs ./plugins ./config
$ echo -e "AIRFLOW_UID=$(id -u)" > .env

# Database Initialization
$ docker compose up airflow-init

# CLI 도와주는 스크립트
$ curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.7.0/airflow.sh'
$ chmod +x ./airflow.sh
```

| Directory | Description                                                          |
|:----------|:---------------------------------------------------------------------|
| ./dags    | DAG files 넣는 곳                                                       |
| ./logs    | task excution 그리고 scheduler 로그                                       |
| ./config  | custom log parser 또는 airflow_local_settings.py 로 cluster policy 를 수정 |
| ./plugins | custom plugins 을 여기에다가 설치                                            |


## Run Airflow

실행은 다음과 같이 합니다.

```bash
$ docker compose up
```
[http://localhost:8080](http://localhost:8080) 으로 접속. <br>
초기 ID, 비번은 둘다 airflow.


## PyCharm 설정은 다음과 같이 합니다.

<img src="{{ page.asset_path }}airflow-docker-01.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">




## CLI Commands 

패턴은 다음과 같습니다.<br>

>> docker compose run \<service\> 명령문

예를 들어서 `airflow-woker` 서비스에서 `airflow info` 명령어를 사용하고 싶으면 다음과 같습니다. 

```bash
docker compose run airflow-worker airflow info
```

주요 명령어는 다음과 같습니다. 

| Category | Command                                       | Description                                     |
|:---------|:----------------------------------------------|:------------------------------------------------|
| DB       | airflow db init                               | DB 초기화                                          |
|          | airflow db reset                              | 기존 데이터 날리고 초기화                                  |
|          | airflow db shell                              | database 에 접속                                   |
| DAG      | airflow dags list                             | dag_id, filepath, owner, paused 등에 대한 정보를 얻습니다. |
|          | airflow dags test [dag name] [execution date] | 테스트                                             |
|          | airflow dags test my_dag now                  |                                                 |
| TASK     | airflow tasks list [dag name] --tree          | 해당 dag 에 대한, tasks 들을 트리구조로 출력합니다.              |



위에서 다운로드 받은 airflow.sh 스크립트를 이용해서 정보를 CLI 도 가능합니다. 

```bash
$ ./airflow.sh info
$ ./airflow.sh bash
$ ./airflow.sh python
```


## Airflow Docker Build

airflow docker 를 수정하거나, 새로운 python package 또는 apt package를 추가해야 할때 docker 를 새로 만들어야 합니다. <br> 
다음은 예제 코드로서 certificate 문제까지 해결한 것입니다. <br>
`/etc/ssl/cert.pem` 을 통해서 certificate이 필요한 것으로 가정했고, 빌드전 복사해놔야 합니다. <br>
필요한 파일은 다음과 같습니다. 

 - requirements.txt
 - cert.pem (/etc/ssl/cert.pem 에서 복사합니다.)
 - DockerFile (아래와 같이 만듭니다)

```yaml
FROM apache/airflow:2.7.0
USER root
COPY cert.pem /etc/ssl/cert.pem
COPY requirement.txt
RUN apt-get update && apt-get install -y ca-certificates
RUN update-ca-certificates
USER airflow
RUN pip config set global.cert /etc/ssl/cert.pem
RUN pip install --no-cache-dir \
  --trusted-host pypi.python.org \
  --trusted-host pypi.org \
  -r /requirements.txt
```


## XCom - Data between tasks

XCom은 tasks 사이에 작은 데이터 또는 메타 데이터를 서로 교환할수 있도록 해줍니다.<br>
다음의 특징을 갖고 있습니다. 

- pull, push 를 사용하며, 실제 데이터는 데이터베이스에 저장이 됩니다. (Postgre)
- task 에서 리턴시, 해당 값은 자동으로 XCom 으로 push가 됩니다. (특히 PythonOperator)
- 데이터 베이스에 따라서 XCom사용시 용량 제한이 걸려 있습니다. 
  - Postgres: 1Gb
  - SQLite: 2Gb
  - MySQL: 64KB
- Custom XCom 사용시 Airflow Database 뿐만 아니라, S3, GCS, HDFS 까지도 사용 가능합니다.

다음은 XCom 예제 입니다.

```python
from airflow.decorators import dag, task
from airflow.models import Param
from airflow.operators.empty import EmptyOperator
from pendulum import datetime

default_args = {"start_date": datetime(2021, 1, 1)}


@dag(schedule="@daily", default_args=default_args, catchup=False,
     params={'name': Param('Anderson', type='string', description='Customer Name'),
             'age': Param(30, type='integer', description='Customer Age')})
def xcom_taskflow_dag():
    start_task = EmptyOperator(task_id='start_task')

    @task
    def get_customer_name(**kwargs):
        params = kwargs['params']
        name = params['name']
        age = params['age']
        return {'customer_name': name, 'age': age}

    @task
    def print_customer_name(**kwargs):
        customer = kwargs['customer']
        print('Customer Name:', customer['customer_name'])
        print('Customer Age :', customer['age'])

    start_task >> print_customer_name(customer=get_customer_name())


xcom_taskflow_dag()
```