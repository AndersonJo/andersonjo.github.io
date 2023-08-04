---
layout: post
title:  "Java Spark for ETL"
date:   2022-10-14 01:00:00
categories: "spark"
asset_path: /assets/images/
tags: ['java', 'spark']
---


# 1. Installation

## 1.1 Spark Installation

https://spark.apache.org/downloads.html 링크에 들어가서 spark-3.4.1-bin-hadoop3.tgz 를 눌러서 다운로드 받습니다. 

<img src="{{ page.asset_path }}spark-java-01.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

```bash
$ tar -zxvf spark-3.4.1-bin-hadoop3.tgz
$ mv spark-3.4.1-bin-hadoop3 ~/app/
$ ln -s ~/app/spark-3.4.1-bin-hadoop3 ~/app/spark
```

.bashrc 파일을 열어서 다음을 추가 합니다. 

```bash
export SPARK_HOME=/home/anderson/app/spark
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH
export PATH=$SPARK_HOME/bin:$SPARK_HOME/python:$PATH
```




# 2. Java Spark ETL

## 2.1 Gradle Configuration

build.gradle 파일에 다음의 스파크 라이브러리를 추가 합니다. <br>
중요한건 `spark-hive_2.13` 을 추가해야지 enableHiveSupport() 함수를 사용할 수 있습니다. 

```bash
dependencies {
    implementation group: 'org.apache.spark', name: 'spark-sql_2.13', version: '3.3.1'
    implementation group: 'org.apache.spark', name: 'spark-hive_2.13', version: '3.4.1'
}
```

## 2.2 Parquet in Hive SQL


## 2.2.1 초기화

다음과 같이 spark session을 초기화 합니다.<br>
.enableHiveSupport() 를 해야지 SQL사용이 가능합니다. 


```java
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

<생략> 
SparkSession spark = SparkSession.builder()
    .master("local[2]")
    .appName("Spark Example")
    .config("spark.ui.port", "4050")
    .config("spark.driver.memory", "2G")
    .config("spark.executor.memory", "4G")
    // .config("spark.sql.warehouse.dir", "/Users/anderson/hive-warehouse")
    // .config("spark.driver.extraJavaOptions", '-Dderby.system.home=/Users/anderson/metastore_db")
    .enableHiveSupport()
    .getOrCreate();
```

### 2.2.2 간단한 SQL

간단한 SQL은 다음과 처리 가능합니다. 

```java
// .show() 를 사용하면 결과를 standard output 으로 출력
spark.sql("show databases").show();
spark.sql("use default");
spark.sql("select current_database()");
```

```bash
+---------+
|namespace|
+---------+
|  default|
+---------+
```

### 2.2.3 Parquet 파일 불러오기

Parquet Data 를 불러오는 것은 다음과 같이 합니다. 

```java
// /data 디렉토리안에 parquet 파일들이 존재함
String dataPath = String.valueOf(this.getClass().getResource("/data"));
Dataset<Row> data = spark.read().parquet(dataPath);
data.show(5);
```

```bash
+-------------------+---+----------+---------+--------------------+------+--------------+----------------+------------+---------+---------+--------------------+--------+
|  registration_dttm| id|first_name|last_name|               email|gender|    ip_address|              cc|     country|birthdate|   salary|               title|comments|
+-------------------+---+----------+---------+--------------------+------+--------------+----------------+------------+---------+---------+--------------------+--------+
|2016-02-03 16:55:29|  1|    Amanda|   Jordan|    ajordan0@com.com|Female|   1.197.201.2|6759521864920116|   Indonesia| 3/8/1971| 49756.53|    Internal Auditor|   1E+02|
|2016-02-04 02:04:03|  2|    Albert|  Freeman|     afreeman1@is.gd|  Male|218.111.175.34|                |      Canada|1/16/1968|150280.17|       Accountant IV|        |
|2016-02-03 10:09:31|  3|    Evelyn|   Morgan|emorgan2@altervis...|Female|  7.161.136.94|6767119071901597|      Russia| 2/1/1960|144972.51| Structural Engineer|        |
|2016-02-03 09:36:21|  4|    Denise|    Riley|    driley3@gmpg.org|Female| 140.35.109.83|3576031598965625|       China| 4/8/1997| 90263.05|Senior Cost Accou...|        |
|2016-02-03 14:05:31|  5|    Carlos|    Burns|cburns4@miitbeian...|      |169.113.235.40|5602256255204850|South Africa|         |     null|                    |        |
+-------------------+---+----------+---------+--------------------+------+--------------+----------------+------------+---------+---------+--------------------+--------+
only showing top 5 rows
```


### 2.2.3 Hive Temporary Table 사용

Temporary Table 을 만들고 SQL처리는 다음과 같이 합니다. 

```java
// SQL 을 사용할수 있도록 만듬
data.createOrReplaceTempView("users");
Dataset<Row> countryData = spark.sql("SELECT country, count(*) FROM users GROUP BY country");
countryData.show();
```

```bash
+--------------------+--------+
|             country|count(1)|
+--------------------+--------+
|              Russia|     310|
|            Paraguay|       8|
<생략>
|         Afghanistan|      21|
+--------------------+--------+
only showing top 20 rows
```


### 2.2.3 Filter 함수

아래코드처럼 column 을 지정해서 데이터를 가져올 수 있습니다. 

```java
Dataset<Row> countries = data
    .filter(data.col("country")
    .equalTo("Japan"));
countries.show();
```

또는 lambda 함수를 써서 만들수도 있습니다. <br>

```java
Dataset<Row> salaries = data.filter((Row row) -> {
    Double salary = row.getAs("salary");
    if (salary == null) {
        return false;
    }
    return salary > 200_000.;
});
```




# 3. Spark must-know things

## RDD, DataFrame, and DataSet

|               | RDD   | DataFrame | Dataset |
|:--------------|:------|:----------|:--------|
| Spark Version | 1.0   | 1.3       | 1.6     |
| Year          | 2011  | 2013      | 2015    |
| Scheme        | No    | Yes       | Yes     |
| API Level     | Low   | High      | High    |



**RDD**
 - Resilient Distributed Dataset
 - 중요 포인트는 transformation methods (map, filter, reduce 함수) 리턴은 다시 동일한 RDD 형태를 리턴한다. 
 - collect() 또는 saveAsObjectFile() 같은 함수가 실행되기 전까지 실제 실행되는 것은 아니다. 

```java
rdd.filter(_.age > 21)
```

**DataFrame**

- 1.3에서 제공되기 시작했고, 속도 그리고 scalability 가 주요 포인트이다. 
- scheme 그리고 serialization을 통해서 노드 사이의 데이터 통신을 더 빠르게 하였다.
- 주로 Scala 에서 많이 활용되며, Java는 제한적인 부분들이 있다.
- RDD가 low level 로서 모든 데이터를 처리 가능하다면, DataFrame 은 tabular dataset 만 처리 가능하다
- 약점은 compile-time type safety 이다. (여러차례의 transformations 그리고 aggregations 거치다 보면 에러날 확률이 높다)


```
// runtime때 문제 나기 쉽다
df.filter("age > 21")
```

**DataSet**
- 1.6에서 도입되었으며, Encoder 를 도입하여, JVM objects 그리고 Spark 내부의 binary format 에서의 translate 를 담당한다. 
- Java, Scala 둘다 동작 잘 됨
- type safety 가 더 좋아졌음으로, DataFrame 에서 나오는 문제는 해결되었지만, Type Casting 에 대한 코드를 더 작성해야 한다. 
- 솔까.. 귀찮음. 

```java
// runtime 때 문제 날수 있는 부분을 해결한다
dataset.filter(_.age < 21)
```