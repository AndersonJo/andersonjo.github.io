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

Cassandra 실행도 시켜봅니다. 

```
$ cassandra -f
```

버젼 확인및 접속

```bash
$ cqlsh --version
cqlsh 6.1.0

# 접속
$ cqlsh localhost 9042
```

## 1.2 Installing only CQLSH

아래와 같이 하면 된다고 하는데, 저는 안되서 그냥 위에꺼 전체 설치 했습니다.

```bash
$ pip install cqlsh 
```

# 2. Cassandra Quick Reference 

## 2.1 Node Status

현재 노드 상태 확인
```
$ nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address    Load        Tokens  Owns (effective)  Host ID                               Rack 
UN  127.0.0.1  205.33 KiB  16      100.0%            36cbcf5a-4753-4491-8fec-2dd168613512  rack1
```

그외
- nodetool info : 각종 시스템 정보
- nodetool ring: 노드가 소유한 토큰의 분포 확인 가능


## 2.2 Basic CQL

**Show databases / tables 같은거**
```
# 기본적인 것
desc keyspaces
desc tables
```

**KeySpace | Table 생성**

```sql
--  KeySpace 생성
CREATE KEYSPACE IF NOT EXISTS my_keyspace
    WITH replication = {
       'class': 'SimpleStrategy', 
       'replication_factor': 1};
```
- SimpleStrategy: 단일 데이터센터 내에서 데이터 복제. 여러개 데이터 센터의 경우 NetworkTopologyStrategy 사용
- Replication Factor: 복제본 갯수. 3이라면 클러스터내의 노드중 3개에 데이터가 복제


NetworkTopologyStrategy 사용시 다음과 같이 생성 가능.
dc1에 3개 복제하고, dc2에 2개 복제

```sql
CREATE KEYSPACE IF NOT EXISTS example_keyspace 
WITH replication = {
    'class': 'NetworkTopologyStrategy', 
    'dc1': 3, 
    'dc2': 2
};
```

# 3. Java + Spark Example


```java
package ai.incredible.cassandra;

import com.datastax.oss.driver.api.core.CqlSession;
import com.datastax.oss.driver.internal.core.metadata.DefaultEndPoint;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.sql.Timestamp;
import java.util.Arrays;
import java.util.Date;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SparkTest {

	@Test
	public void sparkTest() {
		SparkConf conf = new SparkConf()
			.setAppName("Local Spark Example")
			.setMaster("local[2]")
			.set("spark.cassandra.connection.host", "127.0.0.1");

		SparkSession spark = SparkSession.builder()
			.config(conf)
			.getOrCreate();

		try (CqlSession session = CqlSession.builder()
			.addContactEndPoint(
				new DefaultEndPoint(new InetSocketAddress("localhost", 9042)))
			.withLocalDatacenter("datacenter1")
			// .withAuthCredentials()
			.build()) {
			String createKeySpace = "CREATE KEYSPACE IF NOT EXISTS my_keyspace "
				+ "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};";

			String createTable =
				"CREATE TABLE IF NOT EXISTS my_keyspace.users ("
					+ "uid UUID PRIMARY KEY, "
					+ "name text, "
					+ "age int, "
					+ "married boolean,"
					+ "created_at timestamp);";

			System.out.println(createTable);
			session.execute(createKeySpace);
			session.execute(createTable);

		}

		// 데이터 생성
		StructType schema = DataTypes.createStructType(new StructField[] {
			DataTypes.createStructField("uid", DataTypes.StringType, false),
			DataTypes.createStructField("name", DataTypes.StringType, false),
			DataTypes.createStructField("age", DataTypes.IntegerType, false),
			DataTypes.createStructField("married", DataTypes.BooleanType, false),
			DataTypes.createStructField("created_at", DataTypes.TimestampType, false)
		});

		Timestamp timestamp = new Timestamp(new Date().getTime());
		Dataset<Row> userData = spark.createDataFrame(Arrays.asList(
			RowFactory.create(UUID.randomUUID().toString(), "Anderson", 40, true, timestamp),
			RowFactory.create(UUID.randomUUID().toString(), "Alice", 25, false, timestamp),
			RowFactory.create(UUID.randomUUID().toString(), "Yoona", 21, false, timestamp)
		), schema);

		userData.write()
			.format("org.apache.spark.sql.cassandra")
			.mode(SaveMode.Append)
			.option("keyspace", "my_keyspace")
			.option("table", "users")
			.save();

		// 데이터 가져오기
		Dataset<Row> df = spark.read()
			.format("org.apache.spark.sql.cassandra")
			.option("keyspace", "my_keyspace")
			.option("table", "users")
			.load();

		assertTrue(df.count() >= 3);
		Row andersonRow = df.filter("name = 'Anderson'").first();
		assertEquals(40, (int) andersonRow.getAs("age"));
		assertEquals(true, andersonRow.getAs("married"));

		df.show();
	}
}

```