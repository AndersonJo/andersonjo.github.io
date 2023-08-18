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