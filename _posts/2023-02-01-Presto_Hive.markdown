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


# 2. Hive Quick References 

## Create External Table 

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

## INSERT OVERWRITE TABLE

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