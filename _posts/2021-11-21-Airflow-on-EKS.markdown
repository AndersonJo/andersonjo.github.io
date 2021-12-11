---
layout: post 
title:  "Airflow on EKS"
date:   2021-11-21 01:00:00 
categories: "airflow"
asset_path: /assets/images/ 
tags: []
---

<header>
    <img src="{{ page.asset_path }}airflow_01.png" class="center img-responsive img-rounded img-fluid">
</header>


# 1. Installation 

## 1.1 EKS & ECR Login

**EKS Cluster**는 이미 생성되어 있는 것으로 가정하겠습니다.<br>
만약 MFA걸려 있으면 `aws-mfa` 로 인증뒤, EKS Cluster에 로그인 합니다.

{% highlight bash %}
$ aws eks --region <ap-northeast-2> update-kubeconfig --name <cluster_name>
$ kubectl cluster-info
$ kubectl config get-contexts
{% endhighlight %}

**ECR**도 이미 생성되어 있는 것으로 가정하겠습니다. <br>
아래 코드를 통해서 Docker Login을 합니다. 

{% highlight bash %}
$ aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {aws_account_id}.dkr.ecr.{region}.amazonaws.com
{% endhighlight %}


## 1.2 Helm Chart

아래와 같이 설치를 합니다. 

{% highlight bash %}
# Namespace 생성 
$ kubectl create namespace airflow

# Airflow Helm Repo 추가
$ helm repo add apache-airflow https://airflow.apache.org
$ helm repo update
$ helm search repo airflow
NAME                  	CHART VERSION	APP VERSION
apache-airflow/airflow	1.3.0        	2.2.1

# Chart를 설치합니다. 매우 오래걸립니다.
# helm install <RELEASE_NAME> apache-airflow/airflow --namespace <NAMESPACE>
$ helm install airflow apache-airflow/airflow --namespace airflow --debug
$ helm ls -n airflow

# 터미널 하나 더 열고 다음 명령어로 pods이 잘 생성되고 있는지 확인합니다. 
$ kubectl get pods -n airflow
$ helm list -n airflow
{% endhighlight %}


## 1.3 주요 접속 경로 

설치를 다하게 되면 위에 있는 것처럼 Airflow에 접속할 수 있습니다.

- Webserver: `kubectl port-forward svc/airflow-webserver 8080:8080 --namespace airflow`
  - default Username: `admin`kubectl cluster-info
  - default Password: `admin`
- Postgre Connection
  - default Username: `postgres`
  - default Password: `postgres`
  - Port: `5432`
- Dashboard: `kubectl port-forward svc/airflow-flower 5555:5555 --namespace airflow`

- **Secret Key**는 다음과 같이 얻습니다.

{% highlight bash %}
$ echo Fernet Key: $(kubectl get secret --namespace airflow airflow-fernet-key -o jsonpath="{.data.fernet-key}" | base64 --decode)
{% endhighlight %}

**웹서버 접속**은 다음과 같이 합니다. 

{% highlight bash %}
$ kubectl port-forward svc/airflow-webserver 8080:8080 --namespace airflow
{% endhighlight %}

<img src="{{ page.asset_path }}airflow_webserver.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">















# 2. Production Airflow

## 2.1 values.yaml 

먼저 values.yaml 을 파일로 다운받습니다.<br>
이후 values.yaml파일 안에서 모든 설정들을 하시면 됩니다.

{% highlight bash %}
$ mkdir my-airflow-project && cd my-airflow-project
$ mkdir dags  # put dags here
$ helm show values apache-airflow/airflow > values.yaml
{% endhighlight %}

## 2.2 Webserver Secret Key

Static Webserver Secret Key 를 설정해두면, Chart로 디플로이시에 Airflow Components들은 오직 필요할때만 restart하게 됩니다.<br>

### 2.2.1 Python으로 Secret Key 생성

첫번째 방법은 Python으로  Secret Key 를 생성해서 넣어주는 방식입니다. 

{% highlight python %}
$ python3 -c 'import secrets; print(secrets.token_hex(16))'
{% endhighlight %}

values.yaml 파일을 열어서 아래와 같이 설정합니다.

{% highlight yaml %}
webserverSecretKey: <secret key>
{% endhighlight %}

### 2.2.2 Kubernetes Secret Key 사용

두번째 방법은 Kubernetes Secret을 사용하는 방법입니다.<br> 
values.yaml 파일을 열어서 아래와 이름을 지정합니다.

{% highlight yaml %}
webserverSecretKeySecretName: airflow-webserver-secret-key
{% endhighlight %}

{% highlight bash %}
# 먼저 기존 secret 삭제
# kubectl delete secrets airflow-webserver-secret-key

# Secret Key 생성
$ kubectl create secret generic airflow-webserver-secret-key -n airflow \
    --from-literal="webserver-secret-key=$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
    
$ kubectl get secrets
NAME                           TYPE                                  DATA   AGE
airflow-webserver-secret-key   Opaque                                1      41s
{% endhighlight %}


## 2.3 Adding DAGs

dags 디렉토리에 DAG파일을 추가시키면 됩니다.

 - 반드시 pip install 은 `USER airflow` 이후에 와야 합니다.

{% highlight bash %}
cat <<EOM > Dockerfile
FROM apache/airflow
COPY . .
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends vim 
USER airflow
RUN pip install -r requirements.txt
EOM
{% endhighlight %}

dags/simple.py 예제로 생성합니다. 

{% highlight bash %}
cat <<EOM > dags/simple.py
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

with DAG(
        dag_id='simple_example',
        schedule_interval='0 0 * * *',
        start_date=datetime(2021, 1, 1),
        catchup=False,
        dagrun_timeout=timedelta(minutes=60),
        tags=['example', 'example2'],
        params={"example_key": "example_value"},
) as dag:
    run_this_last = DummyOperator(
        task_id='run_this_last',
    )

    # [START howto_operator_bash]
    run_this = BashOperator(
        task_id='run_after_loop',
        bash_command='echo 1',
    )
    # [END howto_operator_bash]

    run_this >> run_this_last

    for i in range(3):
        task = BashOperator(
            task_id='runme_' + str(i),
            bash_command='echo "{{ task_instance_key_str }}" && sleep 1',
        )
        task >> run_this

    # [START howto_operator_bash_template]
    also_run_this = BashOperator(
        task_id='also_run_this',
        bash_command='echo "run_id={{ run_id }} | dag_run={{ dag_run }}"',
    )
    # [END howto_operator_bash_template]
    also_run_this >> run_this_last

# [START howto_operator_bash_skip]
this_will_skip = BashOperator(
    task_id='this_will_skip',
    bash_command='echo "hello world"; exit 99;',
    dag=dag,
)
# [END howto_operator_bash_skip]
this_will_skip >> run_this_last
EOM
{% endhighlight %}


Docker Build 시키고 배포합니다. 

{% highlight bash %}
$ docker build --tag my-dags:v0.0.1 .
$ docker tag my-dags:v0.0.1 123456489123.dkr.ecr.ap-northeast-2.amazonaws.com/ml-airflow:v0.0.1
$ docker push 123456489123.dkr.ecr.ap-northeast-2.amazonaws.com/ml-airflow:v0.0.1
$ helm upgrade airflow apache-airflow/airflow -f values.yaml --namespace airflow \
    --set images.airflow.repository=123456489123.dkr.ecr.ap-northeast-2.amazonaws.com/ml-airflow \
    --set images.airflow.tag=v0.0.1 \
    --timeout 30m
{% endhighlight %}

port-forward를 통해서 Airflow Webserver에 접속합니다. 

{% highlight bash %}
$ kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow
{% endhighlight %}

아래 그림처럼 simple_example 이 보이면 정상적으로 DAG까지 등록이 된 것입니다.

<img src="{{ page.asset_path }}airflow_simple_dag.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">





## 2.4 Configure Airflow


values.yaml 의 주요 설정 요소들은 다음과 같습니다. <br>
복사 붙여넣기가 아니라.. 각각 따로따로 찾아서 수정해야 합니다.

{% highlight yaml %}
# Airflow Home Directory 위치
airflowHome: /opt/airflow

# Airflow executor
# Options: LocalExecutor, CeleryExecutor, KubernetesExecutor, CeleryKubernetesExecutor
executor: "CeleryExecutor"

# variables.yaml 파일과 연결을 시킵니다. 
extraEnvFrom: |
    - configMapRef:
        name: 'airflow-variables'

{% endhighlight %}


**variables.yaml** 파일

{% highlight bash %}
cat <<EOM > variables.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  namespace: airflow
  name: airflow-variables
data:
  AIRFLOW_VAR_MY_S3_BUCKET: "my_s3_name"
EOM
{% endhighlight %}

variables.yaml 을 Kubenetes에 배포합니다. 

{% highlight bash %}
$ kubectl apply -f variables.yaml
$ kubectl get configmap -n airflow
{% endhighlight %}


values.yaml 을 배포후 확인합니다.

{% highlight bash %}
$ helm upgrade airflow apache-airflow/airflow -f values.yaml --namespace airflow \
    --set images.airflow.repository=123456489123.dkr.ecr.ap-northeast-2.amazonaws.com/ml-airflow \
    --set images.airflow.tag=0.0.1 \
    --timeout 30m
$ kubectl port-forward svc/airflow-webserver 8080:8080 --namespace airflow
{% endhighlight %}


## 2.5 External DB

외부에서 생성하는 Database로 설정해주는 것이 좋습니다.<br>

### 2.5.1 PostgreSQL


PostgreSQL를 Kubernetes에 올리고, Airflow를 설정해주는 것을 해보겠습니다.

{% highlight bash %}
$ helm repo add bitnami https://charts.bitnami.com/bitnami
$ helm install airflow-database bitnami/postgresql -n airflow

$ export POSTGRES_PASSWORD=$(kubectl get secret --namespace airflow airflow-database-postgresql -o jsonpath="{.data.postgresql-password}" | base64 --decode)
$ kubectl run airflow-database-postgresql-client --rm --tty -i --restart='Never' --namespace airflow --image docker.io/bitnami/postgresql:11.14.0-debian-10-r17 --env="PGPASSWORD=$POSTGRES_PASSWORD" --command -- psql --host airflow-database-postgresql -U postgres -d postgres -p 5432
{% endhighlight %}

화면에서 시키는대로 하고, PostgreSQL 에 접속합니다.

{% highlight sql %}
GRANT ALL PRIVILEGES on *.* TO postgres with GRANT OPTION;

CREATE DATABASE airflow with ENCODING 'UTF8' LC_COLLATE = 'en_US.UTF-8' LC_CTYPE = 'en_US.UTF-8';
CREATE USER airflow WITH PASSWORD 'airflow';
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO postgres;
{% endhighlight %}

이후의 접속은 port-forward를 사용해서 접속할수 있습니다.

{% highlight sql %}
$ kubectl port-forward svc/airflow-database-postgresql 5432:5432 -n airflow
$ psql -h localhost -p 5432 -U airflow -d airflow -W
{% endhighlight %}



values.yaml 파일을 수정합니다.<br>
여기서 가장 중요한게 host인데 `<db-servicename>.<namespace>.svc.cluster.local` 이런 형식으로 들어가야 합니다.<br>
service name은 `airflow-database-postgresql` 가 들어가야지 IP가 들어가면 안됩니다.

{% highlight yaml %}
# Don't deploy postgres
postgresql:
  enabled: false

# Airflow database & redis config
data:
  # Otherwise pass connection values in
  metadataConnection:
    user: airflow
    pass: airflow
    protocol: postgresql
    host: airflow-database-postgresql.airflow.svc.cluster.local
    port: 5432
    db: airflow

{% endhighlight %}




수정된 내용으로 업그레이드를 합니다. <br>
아래 명령어는 몇분 이상이 소요 될 수 있습니다. 

{% highlight bash %}
$ helm upgrade --install airflow apache-airflow/airflow -f values.yaml --namespace airflow \
    --set images.airflow.repository=123456489123.dkr.ecr.ap-northeast-2.amazonaws.com/ml-airflow \
    --set images.airflow.tag=0.0.1
$ kubectl port-forward svc/airflow-webserver 8080:8080 --namespace airflow
{% endhighlight %}









### ~~2.5.2 MariaDB Database for External DB~~

<span style="color:red">**일단 MariaDB는 안되는 것 같습니다. 포트연결에서 문제 발생함**</span>

{% highlight bash %}
$ helm repo add bitnami https://charts.bitnami.com/bitnami
$ helm install airflow-mariadb bitnami/mariadb -n airflow

# root 암호
$ echo $(kubectl get secret --namespace airflow airflow-mariadb -o jsonpath="{.data.mariadb-root-password}" | base64 --decode)

# Pad 에 접속
$ kubectl run airflow-mariadb-client --rm --tty -i --restart='Never' --image  docker.io/bitnami/mariadb:10.5.13-debian-10-r18 --namespace airflow --command -- bash

# MariaDB 에 접속
$ mysql -h airflow-mariadb.airflow.svc.cluster.local -uroot -p my_database
{% endhighlight %}

화면에서 시키는데로.. 암호먼저 알아내고 어딘가에 적습니다.<br>
이후 1번, 2번 시키는데로 MariaDB에 접속후 다음과 같이 생성합니다.  

{% highlight sql %}
CREATE USER `root`@`%` IDENTIFIED BY '1234';
GRANT ALL PRIVILEGES on *.* TO `root`@`%` with GRANT OPTION;

CREATE DATABASE airflow CHARACTER SET utf8 COLLATE utf8_unicode_ci;
CREATE USER 'airflow'@'localhost' IDENTIFIED BY 'airflow';
GRANT ALL PRIVILEGES ON airflow.* TO 'airflow'@'localhost';
GRANT ALL PRIVILEGES ON airflow.* TO 'airflow'@'%';
FLUSH PRIVILEGES;
SHOW DATABASES;
{% endhighlight %}

이후에 접속을 좀 더 쉽게 하는 방법은.. port-forward를 써서 접속하는 방법입니다.<br>
이때 3306으로 하면 local mariadb와 충돌되니 3307처럼 다른 주소를 사용합니다. 

{% highlight bash %}
$ kubectl port-forward svc/airflow-mariadb 3307:3306 -n airflow
$ mysql -u root -p -P3307
{% endhighlight %}


values.yaml 파일을 수정합니다.<br>
여기서 가장 중요한게 host인데 `<db-servicename>.<namespace>.svc.cluster.local` 이런 형식으로 들어가야 합니다.<br>
service name은 `airflow-mariadb` 가 들어가야지 IP가 들어가면 안됩니다.

{% highlight yaml %}
# Don't deploy postgres
postgresql:
  enabled: false

# Airflow database & redis config
data:
  metadataConnection:
    user: airflow
    pass: airflow
    protocol: mysql
    host: airflow-mariadb.airflow.svc.cluster.local
    port: 3306
    db: airflow
    sslmode: disable
{% endhighlight %}

수정된 내용으로 업그레이드를 합니다. <br>
아래 명령어는 몇분 이상이 소요 될 수 있습니다. 

{% highlight bash %}
$ helm upgrade --install airflow apache-airflow/airflow -f values.yaml --namespace airflow \
    --set images.airflow.repository=123456489123.dkr.ecr.ap-northeast-2.amazonaws.com/ml-airflow \
    --set images.airflow.tag=0.0.1
{% endhighlight %}


## 2.6 Delete External Database

**삭제**시에는 다음과 같이 합니다.<br>
중요한건 persistent volume 도 동시에 삭제 해야 합니다.

{% highlight bash %}
$ helm delete airflow-mariadb -n airflow

# PVC 삭제 
$ kubectl get pvc -n airflow
$ kubectl delete pvc -n airflow <pvc-mariadb-name> 

# Persistent Volume 삭제
$ kubectl get persistentvolume -n airflow
$ kubectl delete persistentvolume -n airflow <mariadb persistenvolume ID>
{% endhighlight %}


## 2.7 Deployment

