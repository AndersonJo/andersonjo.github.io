---
layout: post
title:  "Protocol Buffer 2.5.0 on M1 Macbook"
date:   2023-12-02 01:00:00
categories: "protobuf"
asset_path: /assets/images/
tags: []
---

겁나 오래된 버젼의 protobuf.. 이걸 아직도 사용하는 당신은.. ㅎㅎ

# Installing Protocol Buffers v2.5.0

https://github.com/protocolbuffers/protobuf/releases/tag/v2.5.0 들어가서 소스 말고, protobuf-2.5.0.zip 다운로드 받습니다.

이후 압축해제 후 platform_macros.h 파일을 수정합니다. 

```bash
$ unzip protobuf-2.5.0.zip
$ cd protobuf-2.5.0
$ vi src/google/protobuf/stubs/platform_macros.h
```

다음과 같이 작성된 문장을 찾습니다. 

```bash
#else
#error Host architecture was not detected as supported by protobuf
```

바로 위에 다음을 추가 합니다. 

```bash
#elif defined(__arm64__)
#define GOOGLE_PROTOBUF_ARCH_ARM 1
#define GOOGLE_PROTOBUF_ARCH_64_BIT 1
```


<img src="{{ page.asset_path }}protobuf-m1.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

이후 설치하면 됩니다. 

```bash
$ ./configure
$ make
$ make check
$ sudo make install
$ protoc --version
```
