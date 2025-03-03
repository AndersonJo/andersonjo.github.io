---
layout: post
title:  "ROS2 - Installation"
date:   2024-11-01 01:00:00
categories: "robotistics"
asset_path: /assets/images/
tags: []
---


# 1. Installation Ros2 Jazzy

## 1.1 ROS 2 version and Gazebo version

아래의 테이블을 보고, version을 맞추는게 좋음.
 - 내 설정
   - Ubuntu: 24.04
   - Python 3.10.16 (그 이상 버젼 안됨. 정확하게 3.10 버젼)
   - Ros2: Humble (Ubuntu 24.04 에서 제공. 하위 버젼 설치 안됨)
   - Gazebo: 


| ROS 2 version | Gazebo version | Branch                                                        | Binaries hosted at                                                                                                                               | 
|---------------|----------------|---------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------| 
| Foxy          | Citadel        | [foxy](https://github.com/gazebosim/ros_gz/tree/foxy)         | https://packages.ros.org [^2]                                                                                                                    | 
| Foxy          | Edifice        | [foxy](https://github.com/gazebosim/ros_gz/tree/foxy)         | only from source [^2]                                                                                                                            | 
| Galactic      | Edifice        | [galactic](https://github.com/gazebosim/ros_gz/tree/galactic) | https://packages.ros.org [^2]                                                                                                                    | 
| Galactic      | Fortress       | [galactic](https://github.com/gazebosim/ros_gz/tree/galactic) | only from source                                                                                                                                 | 
| Humble        | Fortress       | [humble](https://github.com/gazebosim/ros_gz/tree/humble)     | https://packages.ros.org                                                                                                                         | 
| Humble        | Garden         | [humble](https://github.com/gazebosim/ros_gz/tree/humble)     | [gazebo packages](https://gazebosim.org/docs/latest/ros_installation#gazebo-garden-with-ros-2-humble-iron-or-rolling-use-with-caution-)[^1] [^2] | 
| Humble        | Harmonic       | [humble](https://github.com/gazebosim/ros_gz/tree/humble)     | [gazebo packages](https://gazebosim.org/docs/harmonic/ros_installation#-gazebo-harmonic-with-ros-2-humble-iron-or-rolling-use-with-caution-)[^1] | 
| Iron          | Fortress       | [humble](https://github.com/gazebosim/ros_gz/tree/iron)       | https://packages.ros.org                                                                                                                         | 
| Iron          | Garden         | [humble](https://github.com/gazebosim/ros_gz/tree/iron)       | only from source [^2]                                                                                                                            | 
| Iron          | Harmonic       | [humble](https://github.com/gazebosim/ros_gz/tree/iron)       | only from source                                                                                                                                 | 
| Jazzy         | Garden         | [ros2](https://github.com/gazebosim/ros_gz/tree/ros2)         | only from source [^2]                                                                                                                            | 
| Jazzy         | Harmonic       | [jazzy](https://github.com/gazebosim/ros_gz/tree/jazzy)       | https://packages.ros.org                                                                                                                         | 
| Rolling       | Garden         | [ros2](https://github.com/gazebosim/ros_gz/tree/ros2)         | only from source [^2]                                                                                                                            | 
| Rolling       | Harmonic       | [ros2](https://github.com/gazebosim/ros_gz/tree/ros2)         | only from source                                                                                                                                 | 
| Rolling       | Ionic          | [ros2](https://github.com/gazebosim/ros_gz/tree/ros2)         | https://packages.ros.org                                                                                                                         | 

## 1.1 Install Ros2 on Ubuntu


Ubuntu 24.04 에서 Humble 설치 
- Ubuntu 24.04는 noble인데, ROS2 Humble은 jammy 기준 패키지를 가져와야 하므로 위처럼 레포를 jammy로 지정

```bash
# ROS 2 GPG key 등록
$ sudo apt update && sudo apt install curl -y
$ sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

$ sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null <<EOF
deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main
EOF


# ROS Tool 설치
$ sudo apt update
$ sudo apt install ros-dev-tools

# ROS2 설치
$ sudo apt update && sudo apt upgrade
$ sudo apt install ros-humble-desktop

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
source /opt/ros/jazzy/setup.bash
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

Franka 설치

```
$ mkdir -p ~/projects/ros-
$ cd ~/projectscccccccc
$ git clone https://github.com/frankaemika/franka_ros2.git
```


## 1.3 Install Gazebo

```bash
# dependencies
$ sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
$ sudo apt-get update

# Ubuntu 22.04 Harmonic
$ sudo apt-get install gz-harmonic

# 정상 작동 확인
$ gz sim
```

이후 삭제는 이런 식으로

```bash
$ sudo apt remove gz-harmonic && sudo apt autoremove
```


## 1.4 Uninstall on Ubuntu

```bash
$ sudo apt remove ~nros-foxy-* && sudo apt autoremove

# 다음 실행해서 삭제
$ sudo rm /etc/apt/sources.list.d/ros2.list
$ sudo apt update
$ sudo apt autoremove
```




# 2. Move it and Frankaemika 

## 2.1 Move it

```bash
# 작업 공간 설정
$ cd ~/projects/ros-arm-rl/src

$ Move it 클론
$ git clone https://github.com/ros-planning/moveit_resources.git
$ git clone https://github.com/ros-planning/moveit2.git

# dependencies 설치
$ git clone https://github.com/ros-industrial/ros_industrial_cmake_boilerplate.git
$ git clone https://github.com/ros2/ros_testing.git

$ git clone https://github.com/ros-industrial/stomp.git
$ git clone https://github.com/moveit/stomp_moveit.git
$ git clone https://github.com/moveit/moveit_visual_tools.git
$ git clone https://github.com/PickNikRobotics/graph_msgs.git

# Moveit2 에서 사용하는 내부 툴 설치 (jazzy 에서 지원 안함)
$ mkdir ament_cmake_google_benchmark && cd ament_cmake_google_benchmark
$ git init && git remote add origin https://github.com/ament/ament_cmake.git
$ git config core.sparseCheckout true
$ echo "ament_cmake_google_benchmark" >> .git/info/sparse-checkout
$ git pull origin rolling

# Move it 2 설치
$ cd ~/projects/ros-arm-rl
$ source /opt/ros/jazzy/setup.bash
$ colcon build --symlink-install --packages-skip moveit_planners_stomp moveit_planners
```


## 2.1 Frankaemika

독일 팔 로봇

```bash
$ pip install numpy pandas lark-parser

# 작업 공간 설정
$ cd ~/projects/ros-arm-rl/src

# Fran
$ git clone https://github.com/frankaemika/franka_ros2.git
$ git clone --recursive https://github.com/frankaemika/libfranka.git
$ cd libfranka
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
$ cmake --build . --parallel $(nproc)
$ sudo cmake --install .

$ cd ..
$ rosdep update
$ rosdep install --from-paths src --ignore-src -r -y
$ colcon build --symlink-install
```