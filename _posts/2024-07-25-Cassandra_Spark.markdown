---
layout: post
title:  "Cassandra + Spark"
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
in
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

만약 CQLSH 버젼이 높고 -> 서버는 3.11.5 같은 낮은 버젼일 경우 desc 가 작동을 안합니다.<br> 
이 경우에는 다음과 같은 명령어로 가능합니다. 

```
SELECT keyspace_name, table_name FROM system_schema.tables;
```



**KeySpace or Table 생성**

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

## 3.1 Gradle 

 - com.datastax.spark:spark-cassandra-connector_2.12:3.4.1: Spaprk 에서 Cassandra 접속 가능
 - com.datastax.oss:java-driver-core:4.17.0: 다이렉트로 Cassandra DB 에 접속 가능 / Spark 없어도 됨

```
dependencies {
    implementation "com.github.jnr:jnr-posix:3.1.15"
    implementation 'joda-time:joda-time:2.12.7'

    implementation group: 'org.projectlombok', name: 'lombok', version: '1.18.34'
    implementation 'org.apache.spark:spark-core_2.12:3.4.1'
    implementation 'org.apache.spark:spark-sql_2.12:3.4.1'
    implementation 'com.datastax.spark:spark-cassandra-connector_2.12:3.4.1'
//    implementation 'com.datastax.oss:java-driver-core:4.17.0'
    testImplementation platform('org.junit:junit-bom:5.10.0')
    testImplementation 'org.junit.jupiter:junit-jupiter'
}
```


## 3.2 Spark Setup + Adding Data to Cassandra

```java
@BeforeEach
public void setup() {
    SparkConf conf = new SparkConf()
        .setAppName("Local Spark Example")
        .setMaster("local[2]")
        // .set("spark.cassandra.auth.username", "user_id")
        // .set("spark.cassandra.auth.password", "password")
        // .set("spark.cassandra.input.throughputMBPerSec", "1")
        .set("spark.cassandra.connection.host", "127.0.0.1");

    spark = SparkSession.builder()
        .config(conf)
        .getOrCreate();

    addTestData();
}

protected void addTestData() {
    try (CqlSession session = CqlSession.builder()
        .addContactEndPoint(
            new DefaultEndPoint(new InetSocketAddress("localhost", 9042)))
        .withLocalDatacenter("datacenter1")
        // .withAuthCredentials("your_username", "your_password") // 사용자 인증 정보 추가
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
}
```


## 3.3 Spark 로 데이터 다 가져오기


```
public void readAllTable() {
    // 방법 1
    // Spark 에서 전체 데이터를 다 가져오기.
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
```

## 3.4 Spark Cassandra Connector 사용 - 1번

버젼에 따라서 이게 될수도 있고, 2번이 될수도 있음. 
회사에서는 해당 1번은 안되고, 2번이 됐는데, 내 컴퓨터에서는 그 반대. 

```java
public void readThroughCassandraConnector1() {
    CassandraTableScanJavaRDD<CassandraRow> rdd =
        javaFunctions(spark.sparkContext())
            .cassandraTable("my_keyspace", "users")
            .select(column("uid"),
                column("name"),
                column("age"),
                column("married"),
                column("created_at").as("createdAt"),
                writeTime("name").as("writetime"));
    JavaRDD<Row> javaRdd = rdd.map(row -> {
        return RowFactory.create(
            row.getString("uid"),
            row.getString("name"),
            row.getInt("age"),
            row.getBoolean("married"),
            new Timestamp(row.getLong("createdAt")),
            row.getLong("writetime"));
    });

    StructType schema = DataTypes.createStructType(new StructField[] {
        DataTypes.createStructField("uid", DataTypes.StringType, false),
        DataTypes.createStructField("name", DataTypes.StringType, false),
        DataTypes.createStructField("age", DataTypes.IntegerType, false),
        DataTypes.createStructField("married", DataTypes.BooleanType, false),
        DataTypes.createStructField("createdAt", DataTypes.TimestampType, false),
        DataTypes.createStructField("writetime", DataTypes.LongType, false)
    });

    Dataset<Row> dataset = spark.createDataFrame(javaRdd, schema);
    dataset.show();
    System.out.println(dataset);

}
```

## 3.5 Spark Cassandra Connector 사용 - 2번


```java
public void readThroughCassandraConnector2() {
    // Spark Cassandra Connector를 사용해서, 좀더 자세한 정보를 가져오는 방법
    // 회사에서는 됐는데, 지금 여기서는 안됨. select 에서 empty 가 나옴.
    CassandraTableScanJavaRDD<DataBean> rdd = javaFunctions(spark.sparkContext())
        .cassandraTable("my_keyspace", "users", mapRowTo(DataBean.class))
        .select(column("uid"),
            column("name"),
            column("age"),
            column("married"),
            column("created_at").as("createdAt"),
            writeTime("name").as("writetime"));
    JavaRDD<Row> javaRdd = rdd.map(row -> {
        return RowFactory.create(
            row.getUid(),
            row.getName(),
            row.getAge(),
            row.getMarried(),
            row.getCreatedAt(),
            row.getWritetime()
        );
    });
    
    Dataset<Row> dataset = spark.createDataFrame(javaRdd, DataBean.class);
    dataset.show();
}
```

DataBean.java

```
package ai.incredible.cassandra;

import lombok.Data;
import lombok.ToString;

import java.sql.Timestamp;

@Data
@ToString
public class DataBean {
	protected String uid;
	protected String name;
	protected Integer age;
	protected Boolean married;
	protected Timestamp createdAt;
	protected Long writetime;
}
```



## 3.1 CQL Session 으로 Direct Connection


```java
// CQL 로 direct 접속을 해서 데이터를 가져옵니다.
// 해당 방법은 spark.read() 를 사용하는 것이 아니며, 이를 spark 에서 사용시에
// driver 에서 바로 가져오는 것이기 때문에 distributed loading 이 되는 것이 아닙니다.
// Spark 에서 쓰는 것 보다는 따로 CQL 로 접속해야 할때 사용하면 좋은 방법입니다.
try (CqlSession session = CqlSession.builder()
.addContactEndPoint(
	new DefaultEndPoint(new InetSocketAddress("localhost", 9042)))
.withLocalDatacenter("datacenter1")
// .withAuthCredentials("your_username", "your_password") // 사용자 인증 정보 추가
.build()) {

// 중요한점! ALLOW FILTERING 에 끝에 들어갔음.
// Cassandra 에서는 WHERE statement 가 연산량이 많은듯 함.
// 그래서 WHERE 사용시 반드시 뒤에 ALLOW FILTERING 써줘야 함
// 또한 setPageSize 를 통해서 한번에 얼마나 가져올지를 정함
String query = "SELECT name, age, WRITETIME(name) as created_at "
	+ "FROM my_keyspace.users WHERE name='Anderson' ALLOW FILTERING;";
ResultSet resultSet = session.execute(SimpleStatement.builder(query)
	.setPageSize(5).build());

List<Row> rows = new ArrayList<>();
do {
	for (com.datastax.oss.driver.api.core.cql.Row cassandraRow : resultSet) {
		rows.add(RowFactory.create(
			cassandraRow.getString("name"),
			cassandraRow.getInt("age"),
			new Timestamp(cassandraRow.getLong("created_at") / 1000)
		));
	}

} while (!resultSet.isFullyFetched());

StructType schema2 = DataTypes.createStructType(new StructField[] {
	DataTypes.createStructField("name", DataTypes.StringType, false),
	DataTypes.createStructField("age", DataTypes.IntegerType, false),
	DataTypes.createStructField("created_at", DataTypes.TimestampType, false)
});

Dataset<Row> df2 = spark.createDataFrame(rows, schema2);
df2.show();
```