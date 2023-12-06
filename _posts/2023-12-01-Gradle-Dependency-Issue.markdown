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



## 1.2 Dependency Configuration

특정 dependency 를 제외 시키는 것은 다음과 같이 할 수 있습니다. <br>
포인트 부분은 버젼에서 `.*` 이렇게 적어서 모든 버젼을 다 제외시키게 만들었습니다. 

```
shadowJar {
    archiveBaseName.set('my-dependency-example')
    archiveClassifier.set("incredible-ai")

    dependencies {
        exclude(dependency('org.spark-project.hive:hive-exec:.*'))
    }

    setZip64(true)
}
```




# 2. How to Inspect Shadow Jar File

## 2.1 Java Decompiler Project

https://java-decompiler.github.io/


| How to               | Description                                                                                                                                                                         |
|:---------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| spark version check  | `spark-version-info.properties` \n  <img src="{{ page.asset_path }}gradle-dependency-01.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333"> |
| maven dependencies   | `/META-INF/maven` 디렉토리에서 dependencies 확인 가능합니다.                                                                                                                                     |



## 2.2 IntelliJ Plugin - File Expander

File -> Settings -> Plugins -> Search "File Expander" <br> 
위와 같이 검색후 File Expander 를 다운로드 받습니다. 

<img src="{{ page.asset_path }}jar-file-expander-intellij.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

아래와 같이 META-INF/maven 에서 설치된 dependencies 들을 확인 할 수 있습니다. 

<img src="{{ page.asset_path }}jar-file-expander-intellij-02.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

