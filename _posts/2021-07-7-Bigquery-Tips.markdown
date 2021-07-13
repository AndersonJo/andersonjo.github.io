---
layout: post
title:  "Troubleshooting BigQuery"
date:   2021-07-07 01:00:00
categories: "gcp"
asset_path: /assets/images/
tags: ['quota', 'limit', 'error', 'bug']
---

# 1. BigQuery  

## 1.1 BigQuery Quota Errors

 - 메타데이터 조회시 최소 10MB에 해당하는 비용이 나갈수 있습니다. (캐쉬가 안되어 있어서 어떤 쿼리를 돌려서 최소 금액이 잡혀 있음)
 - 자세한 내용은 [링크](https://cloud.google.com/bigquery/docs/information-schema-jobs) 참조 합니다. 
 - BigQuery를 사용중 403 에러를 내놓는 경우가 있으며, response 값을 보면 자세하게 알 수 있습니다.
 
메타데이터를 조회시 정확한 에러를 분석할수 있습니다. <br>
메타데이터 테이블은 아래의 종류가 있습니다.

 - `INFORMATION_SCHEMA.JOBS_BY_*` : BigQuery job 에 대한 쿼리 결과값을 조회가능
 - `INFORMATION_SCHEMA.JOBS_BY_USER` : 현재 유저가 보낸 모든 jobs에 대한 메타데이터 조회
 - `INFORMATION_SCHEMA.JOBS_BY_PROJECT` : 현재 포로젝트내의 모든 jobs에 대한 메타데이터 조회
 - `INFORMATION_SCHEMA.JOBS_BY_FOLDER` : 부모 디렉토리에서 생성한 모든 jobs에 대한 메타데이터 조회
 - `INFORMATION_SCHEMA.JOBS_BY_ORGANIZATION` : 현재 organization에서 생성한 모든 jobs에 대한 메타데이터 조회

쿼리 실행시 INFORMATION_SCHEMA 는 다음의 구조로 만듭니다. 

<div class="center text-center">
`PROJECT_ID`.`region-REGION_NAME`.INFORMATION_SCHEMA.VIEW
</div>

예를 들어서 프로젝트 ID는 project-1234 이고 BigQuery의 위치는 asia-northeast3 일 경우 다음과 같이 생성합니다. <br>
주의할 점은 그냥 asia-northeast3 가 아니고 region- 이 반드시 앞에 붙어야 합니다. 

{% highlight sql %}
SELECT
 job_id,
 creation_time,
 error_result
FROM  `project-1234.region-asia-northeast3.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP, INTERVAL 1 DAY) AND
      error_result.reason IN ('rateLimitExceeded', 'quotaExceeded')
{% endhighlight %}

<img src="{{ page.asset_path }}bigquery-ts-information-schema.png" class="img-responsive img-rounded img-fluid border rounded">
