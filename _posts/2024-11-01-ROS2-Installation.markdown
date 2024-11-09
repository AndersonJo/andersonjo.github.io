---
layout: post
title:  "ROS2 - Installation"
date:   2024-11-01 01:00:00
categories: "robotistics"
asset_path: /assets/images/
tags: []
---


# 1. Installation

## 1.1 Ubuntu

- [Ubuntu Install Debians](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
- Ubuntu 20.04 에 설치해야함

```bash
# ROS 2 GPG key 등록
$ sudo apt update && sudo apt install curl -y
$ sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Repository 를 source list 에 추가
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ROS 설치
$ sudo apt update
$ sudo apt install ros-foxy-desktop python3-argcomplete
$ sudo apt install ros-dev-tools
```

**.bashrc** 추가

```bash
# ROS2
source /opt/ros/foxy/setup.bash
```



**Demo**: publisher 그리고 listener 를 체크하는 데모.<br> 
이게 실행 잘 되면, C++ 그리고 Python API 둘다 잘 작동한다는 의미. 

```bash

$ source /opt/ros/foxy/setup.bash
$ ros2 run demo_nodes_cpp talker

# 터미널 새로 하나 열고, 다음을 실행
$ source /opt/ros/foxy/setup.bash
$ ros2 run demo_nodes_py listener
```

## 1.2 Uninstall on Ubuntu

```bash
$ sudo apt remove ~nros-foxy-* && sudo apt autoremove

# 다음 실행해서 삭제
$ sudo rm /etc/apt/sources.list.d/ros2.list
$ sudo apt update
$ sudo apt autoremove
```