---
layout: post
title:  "PySpark for ETL"
date:   2022-10-14 01:00:00
categories: "spark"
asset_path: /assets/images/
tags: ['pyspark', 'parquet', 'csv', 'sql', 'mariadb']
---

# 1. PySpark ETL

# 1.1 Default Initialization

```python
from matplotlib import pylab as plt
from pyspark.sql import SparkSession


spark = (
    SparkSession.builder.master("local[*]")
    .appName("Sales")
    .config("spark.ui.port", "4050")
    .getOrCreate()
)
```

## 1.2 Read CSV

```python
from pyspark.sql import types as t

data = (
    spark.read.option("inferSchema", True)
    .option("delimiter", ",")
    .option("header", True)
    .csv("vgsales.csv")
)

# type casting
data = data.withColumn("Year", data.Year.cast(t.IntegerType()))

# Set Temporary Table
data.createOrReplaceTempView("sales")

print("n rows:", data.count())
data.show(3)
data.limit(3).toPandas()
```


## 1.3 Missing Values

```python
from pyspark.sql import functions as F

def display_missing_count(df):
    missing_df = df.select(
        [
            F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c)
            for c in df.columns
        ]
    ).toPandas()
    display(missing_df)


display_missing_count(data)
```

<img src="{{ page.asset_path }}pyspark-missing-values.png" class="img-responsive img-rounded img-fluid">

아래는 missing values 를 제거 합니다. 

```python
from functools import reduce

def filter_missing_values(df):
    return df.where(
        reduce(
            lambda a, b: a & b,
            [~F.isnan(col) & F.col(col).isNotNull() for col in df.columns],
        )
    )


data = filter_missing_values(data)
display_missing_count(data)
```

## 1.4 GroupBy 

```python
fig, ax = plt.subplots(1, figsize=(4, 3))
(
    data.groupBy("Genre")
    .count()
    .orderBy(F.col("count").desc())
    .toPandas()
    .plot(x="Genre", y="count", kind="bar", ax=ax)
)
```

<img src="{{ page.asset_path }}pyspark-game-genre.png" class="img-responsive img-rounded img-fluid">


## 1.5 SQL Query

```python
query = """
SELECT *
FROM (SELECT *,
             RANK(Global_Sales) OVER (PARTITION BY Year ORDER BY Global_Sales) as rank_sales
      FROM sales) t
WHERE t.rank_sales = 1
ORDER BY Year
"""
df = spark.sql(query).toPandas()
```

## 1.5 Write to Parquet 

`sales.parquet` 디렉토리가 생성됩니다. 

```python
# mode: overwrite, append (새로운 파일이 만들어짐)
data.write.mode("overwrite").parquet("sales.parquet")
```

```text
└── 01 CSV to Parquet.ipynb
    ├── part-00000-3712a917-89eb-4b60-93f3-9f3973cabb0b-c000.snappy.parquet
    └── _SUCCESS
```