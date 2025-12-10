---
layout: post
title:  "Antigravity Personal Settings"
date:   2025-12-01 01:00:00
categories: "development"
asset_path: /assets/images/
tags: ['sfpt', 'oracle']
---



# 1. Hot Keys

## 1.1 Hot Keys

| Category     | Title          | Hot Key            | Description         |
|:-------------|:---------------|:-------------------|:--------------------|
| Antigravity  | Open Command   | CTRL + SHIFT + P   |                     |
|              | Terminal       | CTRL + `           | Open/focus terminal |

## 1.2 Command

| Category | Command                      | Description          |
|:---------|:-----------------------------|:---------------------|
| Python   | Python: Select Interpreter   | 특정 버젼 Python 선택 가능   |




# 2. Extensions

## 2.1 Python Support in Antigravity

Python 실행하고 하려면 해당 extensions 도 설치해야 함. <br>
실제 Python을 설치하는게 아니라, Python을 실행할수 있도록 도와주는 extension 

아래와 같이 검색

```bash
@category:debuggers Python
```

<img src="{{ page.asset_path }}antigravity-sftp-04.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


## 2.2 Remote-SSH: Connect to SSH Host...

먼저 **~/.ssh/config** 에 다음을 작성

```bash
Host oracle
    HostName 134.185.117.137
    Port 22
    User ubuntu
    IdentityFile C:\Users\anderson\.ssh\id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

Bastion 에서는 다음과 같이 설정

```bash
# Bastion 호스트 서버
Host oracle-bastion
    HostName 1.2.3.4
    Port 22
    User ec2-user
    IdentityFile C:/Users/anderson/.ssh/bastion.pem
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
    
# 실제 내부 서버 정의
Host oracle
    HostName 10.0.1.15
    Port 22
    User ubuntu
    IdentityFile C:/Users/anderson/.ssh/id_ed25519
    IdentitiesOnly yes
    
    ProxyJump oracle-bastion

    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```




## 2.3 FTP/SFTP/SSH Sync Tool

FTP/SFTP/SSH Sync 툴에서 + 를 클릭<br>
여기서 해당 remote를 대표하는 이름을 적어 넣습니다.

<img src="{{ page.asset_path }}antigravity-sftp-01.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


SFTP 선택 (따로 FTP 21을 오픈할 필요없이 22번 SSH로 접속 가능)

<img src="{{ page.asset_path }}antigravity-sftp-02.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

다음을 선택 
 - Real-time submission after saving
 - is this the default configuration

다른거 선택하면 안됨!!

<img src="{{ page.asset_path }}antigravity-sftp-03.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


이후에 sync_config.jsonc 가 나오고 여기서 실제 설정.<br>
다음을 반드시 설정

 - host
 - port
 - privateKeyPath
 - remotePath

```json
{
  "oracle_server": {
    "type": "sftp",
    "host": "134.185.117.137",
    "port": 22,
    "username": "ubuntu",
    "privateKeyPath": "C:/Users/anderson/.ssh/id_ed25519",
    "proxy": false,
    "upload_on_save": true,
    "watch": false,
    "submit_git_before_upload": false,
    "submit_git_msg": "",
    "build": "",
    "compress": false,
    "remote_unpacked": false,
    "delete_remote_compress": false,
    "delete_local_compress": false,
    "deleteRemote": false,
    "upload_to_root": false,
    "distPath": [],
    "remotePath": "/home/ubuntu/projects",
    "excludePath": [],
    "downloadPath": "",
    "downloadExcludePath": [],
    "default": true
  }
}
```

