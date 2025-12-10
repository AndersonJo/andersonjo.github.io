---
layout: post
title:  "Antigravity Personal Settings"
date:   2025-12-01 01:00:00
categories: "development"
asset_path: /assets/images/
tags: ['sfpt', 'oracle']
---



# Hot Keys

| Title                | Hot Key                | Description                             |
|:---------------------|:-----------------------|:----------------------------------------|
| Run                  | Shift + F10            | Run current configuration               |
| Debug                | Shift + F9             |  Debug current configuration            |
| Run Current File     | Ctrl + Shift + F10     | Run the current file directly           |
| Stop                 | Ctrl + F2              | Stop running process                    |
| Step Over            | F8                     | Debug step over                         |
| Step Into            | F7                     | Debug step into                         |
| Step Out             | Shift + F8             | Debug step out                          |
| Resume               | F9                     | Resume program in debug mode            |
| Toggle Breakpoint    | Ctrl + F8              | Add/remove breakpoint                   |
| Search Everywhere    | Double Shift           | Search files, classes, symbols, actions |
| Find in Files        | Ctrl + Shift + F       | Search text in all files                |
| Replace in Files     | Ctrl + Shift + R       | Replace text in all files               |
| Go to File           | Ctrl + Shift + N       | Navigate to file by name                |
| Go to Class          | Ctrl + N               | Navigate to class by name               |
| Go to Symbol         | Ctrl + Alt + Shift + N | Navigate to symbol by name              |
| Go to Declaration    | Ctrl + B               | Go to declaration of symbol             |
| Go to Implementation | Ctrl + Alt + B         | Go to implementation                    |
| Find Usages          | Alt + F7               | Find all usages of symbol               |
| Reformat Code        | Ctrl + Alt + L         | Auto-format code                        |
| Optimize Imports     | Ctrl + Alt + O         | Remove unused imports                   |
| Comment Line         | Ctrl + /               | Toggle line comment                     |
| Block Comment        | Ctrl + Shift + /       | Toggle block comment                    |
| Duplicate Line       | Ctrl + D               | Duplicate current line                  |
| Delete Line          | Ctrl + Y               | Delete current line                     |
| Move Line Up         | Alt + Shift + Up       | Move line up                            |
| Move Line Down       | Alt + Shift + Down     | Move line down                          |
| Rename               | Shift + F6             | Rename symbol (refactor)                |
| Extract Method       | Ctrl + Alt + M         | Extract selection to method             |
| Extract Variable     | Ctrl + Alt + V         | Extract to variable                     |
| Quick Documentation  | Ctrl + Q               | Show documentation popup                |
| Parameter Info       | Ctrl + P               | Show parameter hints                    |
| Terminal             | Alt + F12              | Open/focus terminal                     |
| Project View         | Alt + 1                | Toggle project tool window              |
| Version Control      | Alt + 9                | Toggle VCS tool window                  |
| Recent Files         | Ctrl + E               | Show recent files                       |
| Close Tab            | Ctrl + F4              | Close current editor tab                |
| Navigate Back        | Ctrl + Alt + Left      | Go to previous location                 |
| Navigate Forward     | Ctrl + Alt + Right     | Go to next location                     |

# 2. Extensions

## 2.1 FTP/SFTP/SSH Sync Tool

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
  "oracle": {
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

