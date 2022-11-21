---
layout: post
title: "Apache Solr"
date:  2022-09-25 01:00:00
categories: "solr"
asset_path: /assets/images/
tags: ['java']
---

Reference: [Apache Solr Reference Guide](https://solr.apache.org/guide/solr/latest/getting-started/)

# 1. Installation

## 1.1 Installation 

- [Download](https://solr.apache.org/downloads.html) 링크로 들어갑니다.
- Binary release 를 다운받습니다. (본문에서는 9.0 기준입니다.)
- 압축풀고 끝. 




# 2. Solr 서버 실행/중지 

## 2.1 Solr 서버 실행

두가지 방법으로 실행 가능 합니다.<br> 
1. Standalone Mode -> 다른 옵션 없이 `./bin/solr start` 해줘도 됩니다 
2. Solr Cloud Mode -> 그냥 `-c` 옵션 붙여주고 Zookeeper 연결 시켜주면 됩니다.  



- `./bin/solr start`
  - `-p 8983` : port 지정
  - `-s example/node1/solr` : Solr home directory 지정 (아래에 collection이 생성됨)
  - `-c` : Solr Cloud Mode 실행 (embedded zookeeper 까지 실행)
- Local Solr Console: [http://localhost:8983/solr/](http://localhost:8983/solr/)

일반적인 테스트용 local 실행은 **standalone mode** 도 충분 합니다. 

```bash
# 최소 JAVA 11 이 필요합니다. 
$ cd solr-9.0.0
$ JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
$ ./bin/solr start
```

**Solr Cloud Mode**의 경우는 embedded zookeeper 가 열리고, 두번째 열때는 zookeeper 까지 연결 시켜야 합니다.<br>

```bash
$ mkdir -p example/node1/solr
$ mkdir -p example/node2/solr

# Node 1
$ ./bin/solr start -c -p 8983 -s example/node1/solr
$ ./bin/solr start -c -p 7574 -s example/node2/solr -z localhost:9983

# 두개의 노드가 켜져 있는 것 확인 가능
$ ./bin/solr status

# 서버 모두 내림
$ ./bin/solr stop -all
```

## 2.2 Restart

- `./bin/solr restart`


## 2.2 Solr 서버 중지/삭제

- 서버 중지
  - `./bin/solr stop -all`
- 데이터까지 삭제
  - `rm -rf server/solr/<collection name>/`


# 3. Solr References 

## 3.1 Collection

Collection을 만들게 되면 따로 지정하지 않으면 일반적으로 `server/solr/<Collection>` 위치에 저장이 됩니다.<br>
또한 해당 디렉토리 안에 각종 설정 정보들이 함께 생성이 됩니다. 

**Standalone Mode**

- Collections 생성/삭제
    - `./bin/solr create -c <Collection>`
    - `./bin/solr delete -c <Collection>`


**Cloud Mode**<br>
- ` bin/solr create_collection`
  - `-c`: collection 이름
  - `-s`: shards 갯수
  - `-rf`: replicas 갯수

```bash
$ bin/solr create_collection -c films -s 2 -rf 2
```

## 3.2 Managed-schema or schema.xml

이전 버젼에서는 schema.xml 이라고 했으며, Solr 7 이후로는 managed-schema.xml 을 사용합니다.<br>

- `server/solr/<Collection>/conf/managed-schema.xml` 



# 3. Tutorial 

## 3.1 서버 띄우기

2개의 Solr Servers 를 띄웁니다.

```bash
$ ./bin/solr start -e cloud

# 생략 - 계속 엔터치면 됨

Now let's create a new collection for indexing documents in your 2-node cluster.
Please provide a name for your new collection: [gettingstarted] 
techproducts  <- 입력

# 생략 - 계속 엔터치면 됨

Please choose a configuration for the gettingstarted collection, available options are:
_default or sample_techproducts_configs [_default] 
sample_techproducts_configs <- 입력
```

[http://localhost:8983/solr](http://localhost:8983/solr) 접속해서 확인 가능합니다. 



## 3.2 Index the Techproducts Data 

다양한 종류의 파일들을 indexing을 자동으로 걸어주는 명령어를 Solr 에서 지원하고 있습니다.<br>
JSON, CSV, XML 등등 다양한 파일들을 알아서 인덱싱 합니다. 

```bash
$ ./bin/post -c techproducts example/exampledocs/*
```


## 3.3 Query

확인차 쿼리를 날려 봅니다.<br>
해당 쿼리는 콘솔에서도 해볼 수 있습니다.

```bash
$ curl "http://localhost:8983/solr/techproducts/select?indent=on&q=\*:*" | jq .response.docs[].price
$ curl "http://localhost:8983/solr/techproducts/select?indent=on&q=id:samsung" | jq .response.docs

# fl=<word>,<word> .. 사용해서 어떤값을 리턴할지 지정할수도 있습니다. 
$ curl "http://localhost:8983/solr/techproducts/select?indent=on&q=cat:book&fl=cat,name,price" | jq .response.docs
[
  {
    "cat": [
      "book"
    ],
    "name": "A Game of Thrones",
    "price": 7.99
  },
  {
    "cat": [
      "book"
    ],
    "name": "A Clash of Kings",
    "price": 7.99
  },
... 생략
]
```

## 3.4 Phrase Search 

`q=단어 단어 단어` 이렇게 해서 여러 문구로 검색을 할 수도 있습니다. 


```bash
$ curl "http://localhost:8983/solr/techproducts/select?q.op=AND&q=ddr+memory&fl=name" | jq .response.docs
[
  {
    "name": "CORSAIR ValueSelect 1GB 184-Pin DDR SDRAM Unbuffered DDR 400 (PC 3200) System Memory - Retail"
  },
  {
    "name": "A-DATA V-Series 1GB 184-Pin DDR SDRAM Unbuffered DDR 400 (PC 3200) System Memory - OEM"
  },
  {
    "name": "CORSAIR  XMS 2GB (2 x 1GB) 184-Pin DDR SDRAM Unbuffered DDR 400 (PC 3200) Dual Channel Kit System Memory - Retail"
  }
]
```

