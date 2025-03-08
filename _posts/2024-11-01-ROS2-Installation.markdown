---
layout: post
title:  "ROS2 - Installation"
date:   2024-11-01 01:00:00
categories: "robotistics"
asset_path: /assets/images/
tags: []
---


# 1. Installation Ros2 Hubble

## 1.1 ROS 2 version and Gazebo version

아래의 테이블을 보고, version을 맞추는게 좋음.
 - 내 설정
   - Ubuntu: 22.04 (다른 버젼 이래저래 힘듬. 정확하게 22.04)
   - Python 3.10.16 (다른 버젼 안됨. 정확한 버젼) 
   - Ros2: Humble (Ubuntu 24.04 에서 제공. 하위 버젼 설치 안됨)
   - Gazebo: 

## 1.1 Install Ros2 Humble on Ubuntu 22.04


Ubuntu 22.04 에서 Humble 설치 
- 24.04, 20.04 다 안됨. 정확하게 22.04 에서 설치 됨. dependency 문제 생김

```bash
$ sudo apt install software-properties-common
$ sudo add-apt-repository universe

# ROS 2 GPG key 등록
$ sudo apt update && sudo apt install curl -y
$ sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
$ echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list

# ROS2 설치
$ sudo apt update && sudo apt upgrade
$ sudo apt install ros-humble-desktop

# ROS Tool 설치
$ sudo apt update
$ sudo apt install ros-dev-tools

# 설치 확인
$ ros2 doctor --report

# 버젼 확인
$ echo $ROS_DISTRO
```

**.bashrc** 추가

```bash
$ vi ~/.bashrc
```

```bash
# ROS2
source /opt/ros/humble/setup.bash
```



**Demo**: publisher 그리고 listener 를 체크하는 데모.<br> 
이게 실행 잘 되면, C++ 그리고 Python API 둘다 잘 작동한다는 의미. 

```bash

$ ros2 run demo_nodes_cpp talker

# 터미널 새로 하나 열고, 다음을 실행
$ ros2 run demo_nodes_py listener
```


## 1.2 Install Dependencies 

```bash
# 
$ pip install colcon-common-extensions
```



## 1.3 Install Gazebo Fortress

```bash


# GPG key 등록
$ sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
$ sudo apt-get update
$ sudo apt-get install ignition-fortress

# 정상 작동 확인
$ ign gazebo --versions
```

이후 삭제는 이런 식으로

```bash
sudo apt remove ignition-fortress && sudo apt autoremove
```


## 1.4 Uninstall on Ubuntu

```bash
$ sudo apt remove ~nros-humble-* && sudo apt autoremove

# 다음 실행해서 삭제
$ sudo rm /etc/apt/sources.list.d/ros2.list
$ sudo apt update
$ sudo apt autoremove
```




# 2. MoveIt2 & Frankaemika

## 2.1 Install MoveIt2

Dependencies 설치

```bash
$ pip install empy==3.3.4 numpy pandas lark-parser
```


```bash
# 작업 공간 설정
$ mkdir -p ~/projects/ros-arm-rl/src
$ cd ~/projects/ros-arm-rl/src

# MoveIt2 설치
git clone -b humble https://github.com/ros-planning/moveit2.git
git clone -b humble https://github.com/ros-planning/moveit_msgs.git
git clone -b humble https://github.com/ros-planning/moveit_resources.git
git clone -b humble https://github.com/ros-planning/moveit_task_constructor.git
git clone -b humble https://github.com/ros-planning/moveit2_tutorials.git
git clone -b humble https://github.com/ros-planning/srdfdom.git
git clone -b humble https://github.com/ros-controls/control_msgs.git
git clone -b humble https://github.com/ros-controls/ros2_control.git
git clone -b humble https://github.com/ros-controls/ros2_controllers.git
```


```bash
# 의존성 설치 & 빌드
$ cd ..
$ rosdep update
$ rosdep install -y --from-paths src --ignore-src --rosdistro humble
$ colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```


```bash
# MoveIt2 Demo 실행 -> RViz 가 실행되면 MoveIt2는 정상설치
$ ros2 launch moveit_resources_panda_moveit_config demo.launch.py
$ ros2 launch moveit2_tutorials demo.launch.py
$ ros2 launch moveit_task_constructor_demo demo.launch.py
```




## 2.2 Install Franka Emika

독일 팔 로봇

```bash
# Franka Emika 설치
$ cd src 
$ git clone https://github.com/frankaemika/libfranka.git
$ git clone -b humble https://github.com/frankaemika/franka_ros2.git
```

```bash
# libfranka 별도 수동 컴파일 필요
$ cd libfranka
$ git submodule update --init
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
$ cmake --build .
$ sudo cmake --install .
```

```bash
# franka_semantic_component_interface.hpp 파일 수정
$ cd src/franka_ros2/franka_semantic_components/include/franka_semantic_components/
$ vi franka_semantic_component_interface.hpp

# #include "controller_interface/helpers.hpp" <- 이걸 찾아서 다음과 같이 수정 - 23번째줄
#include "controller_interface/controller_interface.hpp"
```

```bash
$ vi src/franka_ros2/franka_semantic_components/CMakeLists.txt

```


```bash
# 의존성 설치 & 빌드
$ cd ..
$ rosdep update
$ rosdep install -y --from-paths src --ignore-src --rosdistro humble
$ colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```