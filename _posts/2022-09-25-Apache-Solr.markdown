---
layout: post
title:  "Apache Solr"
date:   2022-09-25 01:00:00
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

## 1.2 Starting Solr 

```bash
# 서버 실행
$ cd solr-9.0.0
$ JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
$ ./bin/solr start

# 서버 실행 확인
$ ./bin/solr status
```

솔라 콜솔에 접속합니다. 

[http://localhost:8983/solr/](http://localhost:8983/solr/)

## 1.3 Stop Solr

```bash
# 서버 멈추기
$ ./bin/solr stop -all
```

튜토리얼 처음으로 돌리고 싶으면.. 

```bash
$ rm -rf example/cloud/
```

만약 custom으로 collection을 만들었다면.. 삭제는.. 

```bash
$ rm -rf server/solr/<collection name>/
```

이후 재시작합니다.

# 2. References 

## 2.1 Server 

**Server Start**<br>
- `-c`: collection 이름
- `-s`: shards 갯수
- `-rf`: replicas 갯수

```bash
$ bin/solr create_collection -c films -s 2 -rf 2

```


**Restart Example**<br>
두번째부터는 Zookeeper 에 연결시켜야 함

```bash
# Node 1
$ ./bin/solr start -c -p 8983 -s example/cloud/node1/solr

# Node 2
$ ./bin/solr start -c -p 7574 -s example/cloud/node2/solr -z localhost:9983
```

## 2.2 Collection

```bash
# 만들기
$ ./bin/solr create -c <Collection>

# 삭제
$ ./bin/solr delete -c <Collection>
```



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

