---
layout: post
title:  "MariaDB Bulk Insert with LOAD DATA LOCAL INFILE"
date:   2021-08-01 01:00:00
categories: "sql"
asset_path: /assets/images/
tags: ['infile', 'local', 'pymysql', 'sql']
---

# Introduction

최근에 MariaDB에 테라바이트 단위의 데이터를 넣어야 하는 일이 있었습니다. <br>
아.. 네 뭐.. 요즘 빅쿼리도 있고, HDFS도 있고.. 다양한 빅데이터가 있지만.. <br>
사정상 MariaDB로 빠르게 일을 끝내야 하는 일이 있어서 어쩔수 없이 RDBMS로 처리를 해야 했습니다. 

결론만 말씀드리면.. SQL insert는 대용량 처리에는 매우 느립니다.<br>
빠른 방법은 그런 insert구문 없이 로컬 CSV파일을 RDBMS 로 직접 전달하는 것입니다.<br>
대략 속도는 10~20배 정도 더 빠릅니다. 

# 2. Code 

## 2.1 Installation

PyMySQL 설치를 하면 됩니다. 

{% highlight bash %}
pip install --upgrade PyMySQL 
{% endhighlight %}

## 2.2 Connect

아래에서 가장 중요한건 `local_infile=True` 옵션을 주는 것입니다.<br>
이게 안될 경우 `4166, 'The used command is not allowed because the MariaDB server or client has disabled the local infile capability'`
같은 에러가 생길 수 있습니다. 

{% highlight python %}
import pymysql
from pymysql.constants.CLIENT import LOCAL_FILES

connector = pymysql.Connect(
            user='user',
            passwd='1234',
            host='localhost',
            port=3306,
            db='zeta',
            # client_flag=LOCAL_FILES, # 혹시 안되면 이 옵션도 넣으세요
            local_infile=True)
{% endhighlight %}


## 2.3 Run SQL

아래는 LOAD DATA LOCAL INFILE 의 예제 입니다.<br>
윈도우 환경에서도 backslash 가 아니라 forward slash를 사용해야 합니다. <br>
`@컬럼명` 으로 되어 있는 부분은 해당 CSV의 column과 동일 해야 합니다.<br>
SET 이후에 `컬럼명 = @컬럼명` 부분에서 테이블의 컬렴과 연결시켜 주는 부분입니다. 

아래의 SQL을 돌리면 로컬 파일에 있는 CSV를 직접적으로 Remote DB에 올리게 됩니다.<br>

{% highlight sql %}
LOAD DATA LOCAL INFILE 'C:/data/data.csv'
    INTO TABLE dev.rt_transaction
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
    (@txn_at, @stock_code, @price, @change_price)
    SET `txn_at` = @txn_at, `stock_code` = @stock_code, `price` = @price
{% endhighlight %}


아래는 Python에서의 구현입니다. 

{% highlight python %}
file_path = 'C:/data/data.csv'
table_name = 'haha_table'
sql = f'''
LOAD DATA LOCAL INFILE '{file_path}'
    INTO TABLE {table_name}
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
    (@txn_at, @stock_code, @price, @change_price)
    SET `txn_at` = @txn_at, `stock_code` = @stock_code, `price` = @price
    '''

cursor = connector.cursor()
cursor.execute(sql)
connector.commit()
{% endhighlight %}

