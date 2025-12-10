---
layout: post
title:  "Ollama and Open WebUI"
date:   2024-12-01 01:00:00
categories: "llm"
asset_path: /assets/images/
tags: []
---

# 1. Installation

 - python 3.11 사용해야 함

Ubuntu 20.04 기준이고, 에러가 나서 적음 

## 1.1 Upgrade SQLite3

SQLite3 >= 3.35.0 버젼으로 업글 해야합니다.

```bash
# 이렇게 나와서 업글
$ sqlite3 --version
3.31.1 2020-01-27 19:55:54 3bfa9cc97da10598521b342961df8f5f68c7388fa117345eeb516eaa837balt1

# 업글전 삭제
$ sudo apt remove --purge sqlite3 libsqlite3-dev

# 다운로드 받기
$ wget https://www.sqlite.org/2025/sqlite-autoconf-3480000.tar.gz
$ tar xvf sqlite-autoconf-3480000.tar.gz
$ cd sqlite-autoconf-3480000

# 소스에서 빌드 하고 설치
$ sudo ./configure --prefix=/usr
$ sudo make -j$(nproc)
$ sudo make install

# 확인
$ sqlite3 --version
3.48.0 2025-01-14 11:05:00 d2fe6b05f38d9d7cd78c5d252e99ac59f1aea071d669830c1ffe4e8966e84010 (64-bit)
```

## 1.2 Installation

Ollama 설치

```bash
$ curl -fsSL https://ollama.com/install.sh | sh

# test
$ ollama pull llama3.3 
$ ollama list
$ ollama run llama3.3
```

Open WebUI 설치

```bash
$ pip install open-webui
```

# Run Open WebUI

```bash
$ open-webui serve
```

http://localhost:8080/ 여기로 접속