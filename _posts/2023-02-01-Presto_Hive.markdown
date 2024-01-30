---
layout: post
title:  "Presto/Hive Quick References"
date:   2023-02-01 01:00:00
categories: "sql"
asset_path: /assets/images/
tags: ['sql', 'table']
---


# 1. Presto Quick References

## Create Table ... AS SELECT

```sql
CREATE TABLE IF NOT EXISTS schema.table_name
    WITH (
        format = 'parquet', 
        external_location = 's3://bucket/location/v0',
        partitioned_by = ARRAY['created_date']
      )
AS
    WITH temp_table AS (
        SELECT user_id, created_date
        FROM some.table_name_haha
      )

SELECT * 
FROM temp_table
```


## 2. INSERT INTO ... SELECT ...

```sql
INSERT INTO schema.table_name
WITH temp_table AS (
    SELECT user_id, created_date
    FROM some.table_name_haha
)

SELECT *
FROM temp_table
```

## 3. show create table 

select 문에서 나온 테이블을 생성 쿼리를 만들기 위해서 먼저 그냥 샘플로 테이블을 만들어 줍니다. 

```sql
CREATE TABLE haha.table
AS
    SELECT * FROM example
```

이후에 다음과 같은 명령어로 create table 생성 쿼리를 얻을 수 있습니다. 

```sql
SHOW CREATE TABLE haha.table
```



# 2. Hive Quick References 

## 2.1 Create External Table 

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS haha.created_table
(
    id      bigint, 
    name    string,
    age     int, 
    married boolean,
    score   float
)
    PARTITIONED BY (dt string)
    STORED AS PARQUET
    LOCATION 's3://bucket-name/parquet/location'

```

## 2.2 INSERT OVERWRITE TABLE

위에서 만든 테이블에 데이터를 넣으려면 다음과 같이 합니다.<br>
`WITH` clause 는 때에 따라서 써도 되고 빼도 됩니다.  

```sql
SET hive.mapred.mode = nonstrict;
SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;
SET hive.exec.parallel = true;
SET hive.exec.reducers.max = 1;
SET fs.s3.consistent.throwExceptionOnInconsistency=false;
SET fs.s3.consistent=false;

WITH temp_table1 AS
    (SELECT id,
            married
     FROM <some_db>.<table_name>),
WITH temp_table2 AS
    (SELECT id,
            score
     FROM <some_db>.<table_name>)

INSERT OVERWRITE TABLE haha.created_table PARTITION (dt)
SELECT a.id, 
       a.name, 
       a.age,
       b.married, 
       c.score
FROM <db>.<table_name> 
    JOIN temp_table1 b on a.id = b.id
    JOIN temp_table2 c on a.id = c.id;
```

실행하고 나서 s3 를 보면.. temporary directory 가 만들어짐.. <br>
예를 들어서 `.hive-staging_hive_2023-02-01_12-34-45_1234567890-12345` 이런 temporary 디렉토리가 만들어지고..<br>
완료가 되면 `dt=20230201` 같은 디렉토리로 변경이 됨.


## 2.3 explode! and posexplode!

예를 들어서 row 하나에 array가 존재할때 -> row 형태로 만들때는 explode 를 사용합니다. <br>
explode 과 posexplode 차이는 다음과 같습니다. 

- explode: 각각의 element 를 행 형태로 변형합니다. 
- posexplode: 는 explode와 동일하지만 몇번째 idx 인지도 함께 반환합니다. 


| id   | subjects                      | scores         | 
|:-----|:------------------------------|:---------------|
| 123  | ['math', 'english', 'music']  | [20, 30, 60]   |

이렇게 있을때.. 
- explode: ('math', 20), ('math', 30), ('math', 60), ('english', 20), ('english', 30) ... 이렇게 모든 조합이 나갈수 있습니다. 
- posexplode: explode 와 동일합니다. 다만 where 에서 subject_table.idx = score_table.idx 로 필터링 걸면 동일한 순서만 리턴하도록 만들 수 있습니다.
  - 즉 필터 걸면: ('math', 20), ('english', 30), ('music', 60)  이렇게 만들 수 있습니다.


```sql
SELECT 
    id, 
    subject_table.subject, 
    score_table.score
FROM hive.class_score_table 
    LATERAL VIEW posexplode(subjects) subject_table AS idx, subject
    LATERAL VIEW posexplode(scores) score_table AS idx, score
WHERE subject_table.idx = score_table.idx
```



