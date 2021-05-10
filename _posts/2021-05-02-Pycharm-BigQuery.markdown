---
layout: post
title:  "BigQuery Connection with PyCharm"
date:   2021-05-02 01:00:00
categories: "bigquery"
asset_path: /assets/images/
tags: ['bigquery']
---

# 1. BigQuery Connection with PyCharm 

BigQuery접속을 Pycharm (뭐 또는 Datagrip 이나 여튼 IntelliJ IDE) 에서 접속하는 방법을 알아 봅니다. 

## 1.1 윈도우 시간 설정

방금 2시간 날렸는데 이것 때문입니다.<br>
모든 설정이 정상적으로 해도 시간이 맞지 않는다면 접속이 안됩니다. <br>
내부적으로 시스템 시간에 동기화 되는데.. 컴퓨터가 낡고 늙어서 메인보드 배터리가 나가서 지속적으로 시간이 틀어짐. 

1. 윈도우 검색창에서.. `services.msc` 라고 입력
2. **Window Time** 서비스를 찾아서 더블 클릭
3. `시작유형`을 `자동` 으로 변경, 서비스는 `시작` 클릭. 

또는 메인보드 배터리 교체하면 됨. 

## 1.2 GCP Service Account 생성

Service Account 를 생성합니다. 

<img src="{{ page.asset_path }}bigquery-make-service-account-01.png" class="img-responsive img-rounded img-fluid border rounded">

<img src="{{ page.asset_path }}bigquery-make-service-account-02.png" class="img-responsive img-rounded img-fluid border rounded">

저의 경우는 BigQuery Admin 권한을 주는데.. 일단 여기서는 대충 저렇게 Role을 설정. 



## 1.3 키 생성 for Service Account 

생성된 Service Account를 누르고 KEYS 메뉴로 이동합니다.<br>
여기서 새로운 키를 JSON 형식으로 생성합니다. 

<img src="{{ page.asset_path }}bigquery-make-service-account-05.png" class="img-responsive img-rounded img-fluid border rounded">

<img src="{{ page.asset_path }}bigquery-make-service-account-06.png" class="img-responsive img-rounded img-fluid border rounded">

JSON 파일을 다운로드 받고, 해당 위치는 나중에 PyCharm에서 설정시 필요합니다.

## 1.4 PyCharm 에서 BigQuery 설정

Database 메뉴에서 BigQuery를 생성합니다. 

<img src="{{ page.asset_path }}bigquery-make-service-account-10.png" class="img-responsive img-rounded img-fluid border rounded">

<img src="{{ page.asset_path }}bigquery-make-service-account-11.png" class="img-responsive img-rounded img-fluid border rounded">

1. Connection Type: `Service Account`
2. Host: `www.googleapis.com/bigquery/v2`
3. Port: 443
4. User: 당신의 GCP 계정 이메일. (Service Account 이메일 X)
5. Password: 당신의 GCP 암호
6. Project ID: 해당 GCP Project ID
7. OAuthType: 0 
8. OAuthServiceAcctEmail: 방금 만든 service account의 이메일
9. OAuthPvtKeyPath: 방금 만든 service account의 JSON키 파일의 위치


이렇게 하면 접속 됩니다.<br>
여튼 자세한 내용은 [링크](https://www.jetbrains.com/help/datagrip/bigquery.html) 참조