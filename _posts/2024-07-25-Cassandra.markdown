---
layout: post
title:  "Cassandra"
date:   2024-07-25 01:00:00
categories: "cassandra"
asset_path: /assets/images/
tags: ['cqlsh']
---


# 1. Installation

## 1.1 Installing Cassandra DB + CQLSH

1. [https://cassandra.apache.org/_/download.html](https://cassandra.apache.org/_/download.html) 들어갑니다. 
2. [Latest GA Version](https://www.apache.org/dyn/closer.lua/cassandra/4.1.5/apache-cassandra-4.1.5-bin.tar.gz) 을 다운로드 받습니다. 

아래와 같이 설치 가능합니다. 

```bash
$ wget https://dlcdn.apache.org/cassandra/4.1.5/apache-cassandra-4.1.5-bin.tar.gz
$ tar -zxvf apache-cassandra-4.1.5-bin.tar.gz

# 원하는 장소로 이동
$ mv apache-cassandra-4.1.5 ~/apps/

# 이후 .bashrc (ubuntu) 또는 .bash_profile (mac) 을 설정합니다.
$ vi ~/.bashrc 
```

다음과 같이 내용을 (수정 필요) .bashrc 또는 .bash_profile 에 넣습니다. 

```
# Cassandra
CASSANDRA_HOME=/home/anderson/apps/apache-cassandra-4.1.5
export PATH=$PATH:$CASSANDRA_HOME/bin
```

이후 정상 작동하는지 확인합니다. 

```bash
$ cqlsh --version
cqlsh 6.1.0
```

Cassandra 실행도 시켜봅니다. 

```
$ cassandra -f
```

## 1.2 Installing only CQLSH

아래와 같이 하면 된다고 하는데, 저는 안되서 그냥 위에꺼 전체 설치 했습니다.

```bash
$ pip install cqlsh 
```