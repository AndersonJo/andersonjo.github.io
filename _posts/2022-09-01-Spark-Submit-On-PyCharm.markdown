---
layout: post
title: "Spark Installation on Mac and Debugging on IntelliJ"
date:  2022-09-01 01:00:00
categories: "spark"
asset_path: /assets/images/
tags: ['mac', 'intellij', 'shadowjar', 'jdwp']
---


# 1. Installation

## 1.1 Spark + Snappy 설치

Prerequisites 설치 하기

```bash
# Intel
$ brew install automake libtool wget gcc autoconf cmake gzip bzip2 zlib openssl snappy

# M1
$ brew install gcc autoconf automake libtool cmake gzip bzip2 zlib openssl

# M1 - Snappy 설치 
$ sudo chown -R anderson /usr/local/share/zsh
$ arch -x86_64 /usr/local/bin/brew install snappy
```

https://archive.apache.org/dist/spark/ 여기에서 tgz 로 끝나는 spark 다운로드 받으면 됨

```bash
$ wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
$ tar -xvf spark-3.5.1-bin-hadoop3.tgz

# 원하는 곳으로 이동
$ mv spark-3.5.1-bin-hadoop3 ~/apps/
```

## 1.2 spark-env.sh 설정

spark 가 실행될때마다 뭔가 실행시켜야 되는 것들을 여기에 넣습니다. <br>
custom natives 나 그외 기타 설정들도 여기에 넣으면 됩니다. 

```
# Spark 설치한 곳으로 이동
$ cd ~/apps/spark-3.5.1-bin-hadoop3/conf/
$ cp spark-env.sh.template spark-env.sh
$ vi spark-env.sh
```

아래와 같은 내용을 넣습니다.<br>
마지막에 echo도 넣는데, spark 버젼 여러개 관리시에 내가 뭘 실행시킨건지 확인하려고 넣습니다. 

```
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_271.jdk/Contents/Home
export JAVA_LIBRARY_PATH=<그외 설정>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<그외 추가>
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<그외 추가>
export SPARK_LOCAL_IP="127.0.0.1"
echo "Run Spark 3.5.1"
```






## 1.3 IntelliJ - Big Data Tools

File -> Settings -> Plugins ->  `big data tools` 검색후 .. <br> 
`Big Data Tools` Plugin 을 설치 합니다. 

<img src="{{ page.asset_path }}spark-on-pycharm-01.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

근데 이걸로 Spark Job 실행하면은 debug은 안되고 그냥 돌리게 됩니다.<br>
그래서 이 방법은 추천하지 않습니다. 

## 1.4 Run SparkJob in debug mode

먼저 shadowJar 파일을 생성합니다. 

```
gradlew :your:package:shadowJar 
```

이렇게 생성한 이후에 debug 모드로 spark job을 실행합니다. 

```bash

# 어떤 spark를 돌릴지 설정합니다. (저는 여러 버젼을 동시에 사용해요)
$ SPARK_HOME=/home/anderson/apps/spark-3.5.1-bin-hadoop3

# 다음과 같이 실행합니다. 
$ $SPARK_HOME/bin/spark-submit \
    --master local[2] \
    --deploy-mode client \
    --class ai.incredible.spark.TutorialJob \
    --name SparkTutorialJob \
    --conf spark.driver.extraJavaOptions=-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005 \
    your/shadowjar/file/path/shadowjar_file.jar \
    --config your_custom_config
```

중요한 포인트는 `--conf spark.driver.extraJavaOptions=-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005` 
를 넣는 것입니다. 

이거 안되면, 아래와 같이 합니다. cluster mode 에서도 동작 할 겁니다. 

```bash
SET SPARK_SUBMIT_OPTS=-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005
```

IntelliJ에서 **Remote JVM Debug** 를 다음과 같이 만듭니다.

<img src="{{ page.asset_path }}spark-job-debug-mode.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

실행하면 IntelliJ에서 debugging 모드로 실행 가능합니다. 

