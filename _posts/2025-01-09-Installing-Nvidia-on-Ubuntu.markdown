---
layout: post
title:  "Nvidia Driver on Ubuntu"
date:   2024-01-09 01:00:00
categories: "nvidia"
asset_path: /assets/images/
tags: ['format']
---

## Uninstalling Nvidia Driver

```bash
$ sudo apt-get remove --purge 'nvidia-.*'
$ sudo apt-get remove --purge 'cuda-.*'
$ sudo apt-get autoremove
$ sudo apt autoclean
$ sudo apt-get install ubuntu-desktop
$ sudo rm /etc/X11/xorg.conf
$ sudo nvidia-uninstall
```

## Checking Current Nvidia Driver

현재 설치된 Nvidia 버젼을 확인합니다.

```bash
$ modinfo $(find /usr/lib/modules -name nvidia.ko)
```

## Installing Nvidia Driver 

일단 설치 가능한 버젼을 확인합니다. 
```bash
$ sudo ubuntu-drivers list --gpgpu
```

이후 설치 합니다. 

```bash
$ sudo apt install nvidia-common
$ sudo apt install nvidia-driver-xxx
$ sudo apt install nvidia-dkms-xxx
$ sudo apt install nvidia-settings
```
