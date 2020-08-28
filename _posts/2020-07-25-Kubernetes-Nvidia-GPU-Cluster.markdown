---
layout: post
title:  "Kubernetes with Nvidia GPU Cluster"
date:   2020-07-24 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: []
---

<header>
    <img src="{{ page.asset_path }}k8s-wallpaper.png" class="img-responsive img-rounded img-fluid center">
    <div style="text-align:right">
    <a href="https://learnk8s.io/kubernetes-wallpapers">Kubernetes wallpapers</a>
    </div>
</header>


# 1. Installation and Configuration

## Prerequisites 

먼저 아래의 소프트웨어가 설치되어 있어야 합니다.

 - [Docker](http://incredible.ai/docker/2015/12/02/Docker/)

## 1.1 Nvidia-Docker Installation

Nvidia기반의 Docker를 운영하기 위해서는 [Nvidia Contianer Toolkit](https://github.com/NVIDIA/nvidia-docker)를 설치해야 합니다.<br>
설치전 다음이 필요합니다. (2020년 8월 16일 기준)


 - [<span style="color:red">Nvidia driver 미리 설치 필요</span>](http://incredible.ai/tensorflow/2016/05/04/CUDA-TensorFlow-101/)
 - [<span style="color:red">Docker 19.03 미리 설치 필요</span>](http://incredible.ai/docker/2015/12/02/Docker/) 


<img src="{{ page.asset_path }}k8s-nvidia-docker.png" class="img-responsive img-rounded img-fluid center">

Ubuntu 16.04/18.04/20.04 에서 다음과 같이 설치 합니다. 


{% highlight bash %}
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update 
sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime 
sudo systemctl restart docker
{% endhighlight %}

잘되는지 확인해 봅니다.

{% highlight bash %}
#### Test nvidia-smi with the latest official CUDA image
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi

#### 인터렉티브 테스트
docker run --gpus all -it nvidia/cuda:10.0-base /bin/bash

# Start a GPU enabled container on two GPUs
docker run --gpus 2 nvidia/cuda:10.0-base nvidia-smi

# Starting a GPU enabled container on specific GPUs
docker run --gpus '"device=1,2"' nvidia/cuda:10.0-base nvidia-smi
docker run --gpus '"device=UUID-ABCDEF,1"' nvidia/cuda:10.0-base nvidia-smi

# Specifying a capability (graphics, compute, ...) for my container
# Note this is rarely if ever used this way
docker run --gpus all,capabilities=utility nvidia/cuda:10.0-base nvidia-smi
{% endhighlight %}

## 1.2 IOMMU (Input-Output Memory Management Unit)

IOMMU는 가상화 시스템에 올려진 OS에서 하드웨어를 직접 제어하기 위해서 나온 기술입니다. <br>
문제는 하드웨어가 virtualization 그리고 IOMMU 기술을 제공해야 합니다. <br>

먼저 BIOS 에 들어가서 다음의 Enable시켜야 합니다.

 - Intel: VT-d 
 - AMD: IOMMU

먼저 다음을 설치 합니다.

{% highlight bash %}
sudo apt-get install libvirt-bin bridge-utils virt-manager qemu-kvm ovmf
{% endhighlight %}

`sudo vi /etc/default/grub` 열어서 Intel 또는 AMD에 따라서 다음을 추가 합니다. 

{% highlight bash %}
# 인텔의 경우
GRUB_CMDLINE_LINUX="intel_iommu=on"

# AMD
GRUB_CMDLINE_LINUX="amd_iommu=on iommu=pt"
{% endhighlight %}

이후 grub을 업데이트 해줍니다.

{% highlight bash %}
sudo update-grub
virt-host-validate
{% endhighlight %}


~~그 다음으로 GPU하드웨어 주소를 커널 모듈에 설정해줘야 합니다.~~ <br>
~~먼저 GPU의 주소를 알아냅니다.~~ <br>
~~아래의 예제에서는  `10de:1b81` 그리고 `10de:10f0` 입니다.~~ 

{% highlight bash %}
$ sudo lspci -nn | grep -i nvidia
01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GP104 [GeForce GTX 1070] [10de:1b81] (rev a1)
01:00.1 Audio device [0403]: NVIDIA Corporation GP104 High Definition Audio Controller [10de:10f0] (rev a1)
{% endhighlight %}

~~`sudo vi /etc/modprobe.d/vfio.conf` 에 다음을 추가합니다.~~ 

{% highlight bash %}
options vfio-pci ids=10de:1b81,10de:10f0
{% endhighlight %}

업데이트해주고 reboot합니다.

{% highlight bash %}
sudo update-initramfs -u
{% endhighlight %}

## 1.3 KVM2 (Minikube에서 Nvidia 지원 - Optional)

[KVM](https://help.ubuntu.com/community/KVM/Installation) (Kernel-based Virtual Machine) 은 Linux에서 사용할수 있는 가상화 솔루션입니다. <br>
만약 Minikube로 Nvidia GPU사용하려고 한다면.. KVM2를 설치해야 합니다.<br> 
Minikube를 설치하기전에 Nvidia GPU를 사용하기 위해서는 [KVM을 설치](https://help.ubuntu.com/community/KVM/Installation)해야 합니다.<br>
아래의 명령어는 Ubuntu 18.10 또는 이상의 버젼에서 사용가능합니다. 

{% highlight bash %}
sudo apt-get install qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils

sudo adduser `id -un` libvirt
sudo adduser `id -un` kvm

virsh list --all # 테스트: 아무것도 안나오면 정상
egrep -q 'vmx|svm' /proc/cpuinfo && echo yes || echo no  # 테스트: yes로 나와야 함
{% endhighlight %}


{% highlight bash %}
sudo ls -la /var/run/libvirt/libvirt-sock
sudo chown root:libvirt /dev/kvm

# 여기서 리부트 필요

virt-host-validate # 테스트: 최종 잘되는지..
{% endhighlight %}


## 1.4 Kubernetes Installation

### 1.4.1 Minikube Installation

Minikube는 single-node Kuebernetes cluster이며 VM위에서 돌아가며, local 개발시에 편리하게 사용할 수 있습니다.<br>
그냥 한마디로 공부할때 node하나만 띄워놓고 여러가지 실험해볼수 있다는 뜻입니다. 

설치는 다음과 같이 합니다. 

1. [Docker 설치 on Ubuntu](http://incredible.ai/docker/2015/12/02/Docker/)
2. [Minikube 설치하기](https://kubernetes.io/docs/tasks/tools/install-minikube)

버젼확인을 해봅니다. 

{% highlight bash %}
minikube version
{% endhighlight %}


### 1.4.2 Kubernetes 

 - [Install kubeadm](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/)

다음의 패키지를 설치해야 합니다. <br>
kubeadm을 설치할때 자동으로 버젼이 호환되는 kubelet 또는 kubectl을 설치하지 않습니다. <br> 
버젼이 맞지 않을경우 버그가 발생할수 있습니다. 

 - **kubeadm**: 클러스터를 생성함
 - **kubelet**: 클러스터안의 모든 머신위에서 돌아가며, Pod 그리고 containers등을 시작합니다. 
 - **kubectl**: 클라이언트 패키지
 
{% highlight bash %}
sudo apt-get update && sudo apt-get install -y apt-transport-https curl

curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb http://apt.kubernetes.io/ kubernetes-xenial main
EOF

sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
{% endhighlight %}


## 1.5 k8s-device-plugin

여기까지 했다면 다음이 설치가 되어 있어야 합니다. 

 - Nvidia Driver 
 - [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
 - Kubernetes 
 
k8s-device-plugin 은 다음을 지원합니다.

 - 각 노드안에 있는 GPUs 갯수를 설정해줍니다. 
 - GPUs의 heath를 추적합니다. 
 - GPU 컨테이너를 실행하도록 도와줍니다. 
 
**아래는 모든 GPU Nodes에서 실행이 되야 합니다.**

{% highlight bash %}
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker 
{% endhighlight %}

`sudo vi /etc/docker/daemon.json` 에서는 nvidia runtime을 default runtime으로 설정해야 됩니다. 

{% highlight bash %}
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
{% endhighlight %}

모든 GPU nodes에서 위에 있는 내용을 실행시켰다면.. 다음의 Daemonset을 실행시켜서 GPU 서포트를 실행합니다. 

{% highlight bash %}
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.6.0/nvidia-device-plugin.yml
{% endhighlight %}



## 1.6 Minikube 실행

클러스터 실행은 `--driver=kvm2 --kvm-gpu`로 해줘야 합니다. (Nvidia GPU사용하려면..) <br>
Nvidia GPU필요없으면 드라이버 옵션 없이 `minikube start` 해주면 됩니다.

{% highlight bash %}
minikube start --driver kvm2 --kvm-gpu
kubectl version
{% endhighlight %}

Minikube를 실행하기 위해서는 VM(Docker)이 설치되어 있어야 합니다. <br> 
특정 VM을 명시적으로 선택하려면 `--driver=docker` 이런 옵션을 넣어주면 됩니다.

`minikube start` 가 정상적으로 잘 되었다면 다음을 실행하여, NVIDIA driver가 VM에 설치 되도록 합니다.

{% highlight bash %}
minikube addons enable nvidia-gpu-device-plugin
minikube addons enable nvidia-driver-installer
{% endhighlight %}

테스트는 다음과 같이 하며, `nvidia.com/gpu` 가 보여야 합니다.

{% highlight bash %}
kubectl get nodes -ojson | jq .items[].status.capacity
{% endhighlight %}

만약에 권한과 관련된 warning이 뜨면 다음의 명령어로 권한을 수정해 줍니다.

{% highlight bash %}
sudo chown -R $USER $HOME/.minikube
sudo chgrp -R $USER $HOME/.minikube
sudo chown -R $USER $HOME/.kube
sudo chgrp -R $USER $HOME/.kube
{% endhighlight %}

또는 delete 명령어도 권한 문제에 도움이 될 수 있습니다. 

{% highlight bash %}
sudo minikube delete
{% endhighlight %}

잘되는지 확인은 status를 사용합니다.

{% highlight bash %}
minikube status
kubectl version
{% endhighlight %}


## 1.7 .bash 설정

`~/.bashrc` 에 다음을 추가합니다.  <br>
이후부터는 `kubectl` 명령어가 아니라 그냥 `k` 로도 됩니다. 

{% highlight bash %}
# kubectl autocomplete
source <(kubectl completion bash)

# kubectl shorthand 
alias k=kubectl
complete -F __start_kubectl k
{% endhighlight %}




# 2. Code Snippets  

## 2.1 Deployment 부터  Service 까지 

| No | Name               | Command                                                                 | Description       | 
|:---|:-------------------|:------------------------------------------------------------------------|:------------------|
| 1  | Docker 테스트      | `docker run -p 5000:80/tcp [이미지주소]`                                | 이미지 테스트     |
| 2  | **Deployment**     | `kubectl create deployment [디플로이먼트 이름] --image=[이미지 주소]`   | Kubernetes 배포   | 
| 3  | Port-forward (Pod) | `kubectl port-forward [Pod 이름] 5000:8080`                             | Pod 테스트        |
| 3  | Port-forward (Deployment)  | `kubectl port-forward deployments/[디플로이먼트 이름] 5000:80`  | Deployment 테스트 |
| 4  | Expose             | `k expose deployment/[디플로이먼트 이름] --name [서비스 이름] `<br> `--type=NodePort --port 8080 --target-port 8080` |  |
| 5  | Chrome 에서 확인   | `http://[minikube ip]:[node port]`                                      |                   |

- Expose 시 주의 점
   - Minikube에서는 `--type NodePort` 만 가능 (사용시 external ip가 pending으로 잡힘 -> 안됨)
   - 즉 `--type LoadBalancer` 는 안됨
   - `--port` : 서비스 객체가 리스닝하는 port 
   - `--target-port` : 서비스 객체가 받은 이후 트래픽을 보내야 하는 포트

Example 은 다음과 같습니다. 

{% highlight bash %}
# Docker Test
$ docker run -p 8080:8080 gcr.io/google-samples/kubernetes-bootcamp:v1

# Kubernetes 에서 배포 생성
$ kubectl create deployment test-deployment --image=gcr.io/google-samples/kubernetes-bootcamp:v1

# Port-forward로 확인 ->  http://localhost:8080 에서 확인 
$ kubectl port-forward test-deployment-5b69d9b6c6-m4czw 8080:8080
$ kubectl port-forward deployments/test-deployment 8080:8080

# Expose (Minikube에서는 Nodeport만 허용) 
$ kubectl expose deployment/test-deployment --name test-service --type="NodePort" --port=8080 --target-port=8080

# NodePort 를 찾는다 (ex. 30834)
# http://[minikube ip]:[node port] <- 접속
$ kubectl describe service/test-service 
$ minikube ip
{% endhighlight %}



## 2.2 ConfigMaps

이미지는 동일한데, 환경변수 또는 설정 파일이 다른 경우가 있을수 있습니다. <br>
이 경우 매번 다른 container image를 만들기 어렵기 때문에 디커플링 시켜줄 필요가 있씁니다. <br>
이때 사용하는 것이 ConfigMap 과 Secret 입니다. 

ConfigMap 그리고 Secret 모두 Pod으로 넘겨야 하는데 방식은 두가지가 있습니다. 

- 환경 변수로 넘김 
- 디스크 볼륨으로 마움트 시킴

### 2.2.1 ConfigMap Command 

실무에서는 잘 안씀..<br>
그냥 ConfigMap 뭐 있는지.. 리스트나 값 보는 것만 중요하게 보면 됨

 - `kubectl create configmap [ConfigMap 이름]` : ConfigMap 생성 
 - `kubectl get configmap` : ConfigMap 리스트 출력 
 - `kubectl describe configmap [ConfigMap 이름]` : Key-value 출력  

{% highlight bash %}
$ kubectl create configmap test-config --from-literal=env-mode=production
$ kubectl get configmap
NAME                              DATA   AGE
test-config                       1      12s

$ kubectl describe configmap test-config
<생략>
Data
====
env-mode:
----
production
{% endhighlight %}


### 2.2.2 Volume 에 올리는 방법

redis.conf 파일. 

{% highlight text %}
maxmemory 2mb
maxmemory-policy allkeys-lru 
{% endhighlight %}

redis-pod.yaml 파일.

{% highlight yaml %}
apiVersion: v1
kind: Pod
metadata:
  name: redis
spec:
  containers:
    - name: redis
      image: redis:5.0.4
      command:
        - redis-server
        - "/redis-master/redis.conf"
      env:
        - name: MASTER
          value: "true"
      ports:
        - containerPort: 6379
      resources:
        limits:
          cpu: "0.1"
      volumeMounts:
        - mountPath: /redis-master-data
          name: data
        - mountPath: /redis-master
          name: config
  volumes:
    - name: config
      configMap:
        name: example-redis-config
        items:
          - key: redis-config
            path: redis.conf
{% endhighlight %}





# 3. Quick References

## 3.1 기본 명령어  

**주요 상태 정보**는 다음의 명령어를 사용합니다.

| Object      | Command                           | Options                              |
|:------------|:----------------------------------|:-------------------------------------|
| Cluster     | `kubectl cluster-info`            |                                      |
| Namespace   | `kubectl get namespaces`          |                                      |
| Node        | `kubectl get nodes`               |                                      |
| Pod         | `kubectl get pods`                |                                      |
| Pod         | `kubectl get pods -w`             | `-w` : watch                         |
| Pod         | `kubectl get pods -l app=nginx`   | `-l app=nginx` : 필터링 하는 옵션    |  
| Deployment  | `kubectl get deployments`         |                                      |
| Service     | `kubectl get services`            |                                      |
| ConfigMap   | `kubectl get configmap            |                                      |
| Minikube IP | `minikube ip`                     |                                      |


**Describe**는 다음과 같이 사용합니다. 

| Object      | Command                     |
|:------------|:----------------------------|
| Node        | `kubectl describe node [Node 이름]`   |
| Pod         | - `kubectl describe  pods [Pod 이름]` <br> - `kubectl describe  [Pod 이름]`       |
| Service     | `kubectl describe service [Service 이름]`     |
| Deployment  | `kubectl describe deployment [Deployment 이름]` |


## 3.2 YAML 파일에 대한 명령어   

| What        | Command                           | Options                              |
|:------------|:----------------------------------|:-------------------------------------|
| 생성        | `kubectl -f [파일 이름]`          |  `-f [파일이름]` : 파일 지정         |
| 삭제        | `kubectl delete -f [파일이름]`    |                                      |


