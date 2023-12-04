---
layout: post
title:  "Gradle Dependency & Shadow Jar Inspection"
date:   2023-12-01 01:00:00
categories: "java"
asset_path: /assets/images/
tags: []
---

# 1. Gradle Introduction 

## 1.1 Gradle 101

- dependencies 
  - implementation: 해당 모듈에서만 사용. 다른 모듈에서 사용시 해당 dependency 사용 불가
  - api: 해당 모듈뿐만 아니라, 다른 모듈의 dependency 에도 영향을 줍니다.
  - compileOnly: 컴파일시에만 필요하고, 런타임시에는 사용안됩니다. 
    - 예를 들어서 EMR 클러스터안에 이미 spark가 존재하기 때문에, compileOnly 로 설정하게 되면, jar shadow 만들어질때는 spark가 포함이 안되게 됩니다. 
  - runtimeOnly: 런타임시에만 사용되며, 
    - 예를 들어서 mysql-connector-java 같은 경우 런타임시 실제 접속할때 사용하게 되며, 코딩할때는 JDBC API 를 통해서 코드를 컴파일 합니다.


# 2. How to Inspect Shadow Jar File

## 2.1 Java Decompiler Project

https://java-decompiler.github.io/


| How to               | Description                                                                                                                                                                         |
|:---------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| spark version check  | `spark-version-info.properties` \n  <img src="{{ page.asset_path }}gradle-dependency-01.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333"> |




