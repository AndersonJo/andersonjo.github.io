---
layout: post
title:  "ROS2 in Docker with VS Code - Installation"
date:   2024-11-05 01:00:00
categories: "robotistics"
asset_path: /assets/images/
tags: ['devcontainer', 'vscode']
---


# 1. VSCode

## 1.1 VS Code Reference


| Key                    | Description       |
|:-----------------------|:------------------|
| CTRL + K + O           | 디렉토리를 오픈함         |
| CTRL + SHIFT + P or F1 | 명령어를 찾아서 실행시킬수 있음 |
| CTRL + `               | 터미널 창 오픈          |
| CTRL + SHIFT + X       | extension 설치 및 관리 |
| CTRL + SHIFT + I       | Auto Formatting   |


CTRL + P 누르고 아래 같다 붙이기, IntellIJ 처럼 사용 가능해짐
```bash
https://marketplace.visualstudio.com/items?itemName=k--kato.intellij-idea-keybindings
```


## 1.2 VSCode Settings

**Preferences: Open User Settings (JSON)**

```json
{
  "workbench.tree.indent": 20
}
```


# 2. ROS2 

## 2.1 GUI 테스트
```bash
$ sudo apt install x11-apps
$ xeyes
```



# 1. Tutorial  

## 1.1 .devcontainer 설정

디렉토리 구조

```bash
my-ros2-dev/
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
├── src/
```

**.devcontainer/devcontainer.json**

```json
{
  "name": "ros2-franka-dev",
  "build": {
    "dockerfile": "Dockerfile"
  },
  "runArgs": [
    "--privileged",
    "--network=host",
    "--env", "DISPLAY=${env:DISPLAY}",
    "--volume", "/tmp/.X11-unix:/tmp/.X11-unix",
    "--volume", "${localWorkspaceFolder}/src:/ws/src"
  ],
  "settings": {
    "terminal.integrated.defaultProfile.linux": "bash"
  },
  "extensions": [
    "ms-iot.vscode-ros",
    "ms-vscode.cpptools"
  ],
  "postCreateCommand": "source /opt/ros/humble/setup.bash && colcon build --symlink-install",
  "remoteUser": "root"
}
```

**devcontainer/Dockerfile**

```
FROM ros:humble

# 기본 패키지 설치
RUN apt update && apt install -y \
    python3-colcon-common-extensions \
    ros-humble-moveit \
    ros-humble-moveit-setup-assistant \
    ros-humble-franka-msgs \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# workspace 설정
RUN mkdir -p /ws/src
WORKDIR /ws
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## 1.2 VS Code

1. VS Code에서 ~/ros2-dev 디렉토리 열기
2. F1 (또는 CTRL + SHIFT + P) 누르고 `Reopen in Container` 선택

<img src="{{ page.asset_path }}ros2-vscode-reopen-in-container.png" class="img-responsive img-rounded img-fluid">

3. 이후 컨테이너가 자동으로 빌드되고, 컨테이너 안에서 VS code가 작동함


