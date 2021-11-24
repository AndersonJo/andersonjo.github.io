---
layout: post 
title:  "Incremental Batch Transfer from AWS RDS to BigQuery"
date:   2021-09-22 01:00:00 
categories: "data-engineering"
asset_path: /assets/images/ 
tags: []
---

# 1. Architecture

아래의 그림처럼 먼저 RDS에서 S3로 staging files 을 떨군뒤 -> BigQuery 로 가져오는 형태입니다.<br>
RDS -> S3: AWS Data Pipeline 을 사용하며, 만약 Redshift 사용시 S3거치지 않고 넣을수도 있습니다.

<img src="{{ page.asset_path }}batch_transfer_01.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">




# 2. AWS Data Pipeline

## 2.1 IAM Permission for Users or Groups

User 또는 Groups 에 다음의 권한이 있어야지 Data Pipeline 개발이 가능 합니다.

 - `AWSDataPipeline_FullAccess`

## 2.2 IAM Roles for Data Pipeline

Data Pipeline Role 은 다음 2가지가 필요 합니다. 

1. Data Pipeline Role 생성
   1. IAM -> Role -> Create Role
   2. Data Pipeline 선택 -> Data Pipeline 선택
   3. Name: `DataPipelineRole`
2. EC2 Role for Data Pipeline 생성
   1. IAM -> Role -> Create Role
   2. EC2 Role for Data Pipeline 선택
   3. Name: `DataPipelineForEC2Role`

<img src="{{ page.asset_path }}batch_transfer_02.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">



## 2.3 Data Pipeline

AWS Data Pipeline -> Create Pipeline 선택 합니다.

1. Basic Information
   1. Name: RDS to S3 Daily
   2. Build using a template: `Incremental copy of RDS MySQL Table to S3`
2. Parameters
   1. Last modified column name: `created_at`
   2. RDS MySQL password: `RDS 패스워드`
   3. RDS MySQL table name: 가져오려는 `테이블 이름`
   4. Output S3 folder: `S3 저장 위치` 
   5. RDS MySQL username: `RDS Username`
   6. EC2 instance type: transfer가 시작할때 사용하는 `EC2 Instance` 
   7. RDS Instance ID: RDS에서  `DB identifier` 사용하면 됨
3. Security/Access
   1. IAM Roles에서 생성한 default Roles 두개를 선택합니다. 
   2. `DataPipelineRole` 그리고 `DataPipelineForEC2Role` 선택

## 2.4 Add more Tables 

현재 테이블 1개를 S3로 가져오는 것을 설정했습니다. <br>
하지만 대부분의 케이스는 여러개의 테이블을 가져오는 것 입니다.<br>
테이블을 더 추가하기 위해서 `Edit Pipeline`을 선택합니다. 


`Add -> SQLDataNode 추가` 합니다.<br>
RDS에서 가져오는 것이기 때문에 SQLDataNode를 선택하는 것 입니다. 

<img src="{{ page.asset_path }}batch_transfer_03.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

새로 생성된 DefaultSqlDataNode1은 Schedule (Every 1 hour) 노드 아래에 생성이 됩니다.<br>
수정할 부분은, RDS에 접속한뒤 테이블 데이터를 가져와야 하기 때문에 RdsDatabase 아래에 새로 생성한 노드를 놔야 합니다.<br>
그래프 상에서 마우스 드래그로는 수정이 안되고.. 

1. DefaultSQLDataNode1 노드를 선택
2. `Add an optional field` 버튼을 클릭
3. `Database` 추가
4. Database 옵션에서 `rds_mysql` 선택

<img src="{{ page.asset_path }}batch_transfer_04.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


1. Name: Table2 로 변경합니다. 
2. `Add an optional field` -> `Select Query` 추가
3. `SourceRDSTable 의 Select Query 부분을 복사`해서 `Table2의 Select Query에 붙여넣기` 합니다.
4. 옵션으로 `{myRDSTableLastModifiedCol}` 부분이 현재 created_at

{% highlight sql %}
select * from #{table} 
  where #{myRDSTableLastModifiedCol} >= '#{format(@scheduledStartTime, 'YYYY-MM-dd HH-mm-ss')}' and 
        #{myRDSTableLastModifiedCol} <= '#{format(@scheduledEndTime, 'YYYY-MM-dd HH-mm-ss')}'
{% endhighlight %}

 - `#{table}` : `Name` 에서 지정한 `Table2` 를 의미합니다. 따라서 정확한 테이블 명이 되야 합니다.
 - `@scheduledStartTime` 그리고 `@scheduledEndTime` : Schedule 필들에서 가져온 값
 - `#{}` : 그외 parameters 에서 지정된 값들을 의미 합니다.

<img src="{{ page.asset_path }}batch_transfer_05.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


1. Add -> CopyActivity 를 추가합니다.

<img src="{{ page.asset_path }}batch_transfer_06.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

새로 생성된 CopyActivity는 Schedule 노드 아래에 붙는데.. 이동 시켜줘야 합니다.

1. 새로 생성된 CopyActivity 선택 -> `Input` -> `Table2` 선택
2. `Name`: `CopyTable2` 변경
3. `Add an optional field` -> `Runs on` -> `EC2Instance` 선택
4. `Output` : `Create new: DataNode` 선택 

<img src="{{ page.asset_path }}batch_transfer_07.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

새로 만들어진 DataNode 를 선택후..

1. `Type`: `S3DataNode` 변경
2. `Add an optional field`: `File Path` 추가

File Path는 다음과 같음. 

{% highlight sql %}
#{myOutputS3Loc}/#{format(@scheduledStartTime, 'YYYY-MM-dd-HH-mm-ss')}_table2
{% endhighlight %}