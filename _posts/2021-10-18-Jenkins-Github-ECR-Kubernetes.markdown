---
layout: post 
title:  "Jenkins + GitHub + ECR + Kubernetes"
date:   2021-10-18 01:00:00 
categories: "engineering"
asset_path: /assets/images/ 
tags: []
---


<header>
    <img src="{{ page.asset_path }}coffee_keyboard.jpeg" class="center img-responsive img-rounded img-fluid">
</header>



# 1 EC2 생성 for Jenkins

AWS에서 EC2 생성하는 방법인데.. 너무나 기초적이라서.. 알면 패스해도 됩니다. 

## 1.1 Creating Key-Pair

[https://console.aws.amazon.com/ec2/](https://console.aws.amazon.com/ec2/) 접속후, <br>
`Network & Security -> Key Pairs` 메뉴를 선택하고 Create Key Pair 버튼을 선택합니다.<br>
이름은 적절하게 선택하고, RSA, .pem 을 선택후 생성합니다. 

<img src="{{ page.asset_path }}jenkins-05.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

이후 .pem 파일은 ~/.ssh 로 이동시키고, 400 으로 읽기만 할수 있도록 권한 조정을 합니다.

{% highlight bash %}
$ mv *.pem ~/.ssh/
$ chmod 400 ~/.ssh/zeta.pem
{% endhighlight %}

## 1.2 Creating Security Group

Security Group은 Firewall 역활을 합니다.<br>
`EC2 -> Network % Security -> Security Groups` 누르고, `Create Security Group` 버튼을 누릅니다.

Inbound Rules에서 SSH, HTTP, HTTPS 추가하고, Custom TCP 에서 포트 8080도 추가합니다. 

<img src="{{ page.asset_path }}jenkins-06.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


## 1.3 EC2 Instance 생성

Intance는 `Community AMIs -> Ubuntu` 선택후 20.04 또는 18.04 같은 적절한 버젼을 선택합니다.<br>
저는 20.04 버젼인 `ami-09e67e426f25ce0d7` 인스턴스를 기반으로 생성했습니다. <br>
중요한건 Configure Security Group 설정시, 바로 이전에 만들었던 webserver를 선택합니다. 

<img src="{{ page.asset_path }}jenkins-07.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

이것만 설정하고, 바로 생성합니다.<br>
설정후 SSH 접속은 다음과 같이 합니다. 

Username이 OS마다 다른데, centos, ubuntu, root, ec2-user 등등 OS마다 다릅니다.

{% highlight bash %}
$ ssh -i ~/.ssh/zeta.pem ubuntu@54.167.240.88
{% endhighlight %}

그리고 위에서 Jenkins 설치하듯이 하면 됩니다. 


















# 2. Jenkins

## 2.1 Installation

Ubuntu Package 에서 기본적으로 제공하는 Jenkins의 경우 old version 을 제공하고 있습니다. <br> 
최신 버전의 Jenkins 를 설치하기 위해서는 Jenkins에서 제공하는 패키지를 설치해야 합니다.

Dependencies 설치가 필요한데, 초기 AWS Instance에서 필요합니다. 

{% highlight bash %}
$ sudo apt install ca-certificates
$ sudo apt install openjdk-11-jdk
{% endhighlight %}

Jenkins 설치 ..

{% highlight bash %}
$ wget -q -O - https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo apt-key add -
$ sudo sh -c 'echo deb http://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
$ sudo apt update
$ sudo apt install jenkins
{% endhighlight %}

## 2.2 Starting Jenkins

systemctl 로 시작하고, status를 통해서 상태를 확인합니다. 

{% highlight bash %}
$ sudo systemctl start jenkins
$ sudo systemctl status jenkins
{% endhighlight %}


## 2.3 OpenSSH

{% highlight bash %}
$ sudo apt-get install openssh-server
$ sudo systemctl enable ssh
$ sudo systemctl start ssh
{% endhighlight %}

설치이후 `ssh user@localhost` 같은 명령어로 테스트 해봅니다. 

## 2.4 Configure Firewall (Optional)

아래 코드는 ssh (port 22) 그리고 8080 포트를 여는 명령어 입니다. <br>
AWS는 Security Group에서 하면 됨으로 패스합니다. 

{% highlight bash %}
$ sudo apt-get install ufw
$ sudo ufw enable
$ sudo ufw allow ssh
$ sudo ufw allow 8080
{% endhighlight %}

## 2.5 Setting Up Jenkins

`http://jenkins_server_ip_address:8080` 으로 접속시 다음과 화면이 보이고, 암호를 넣어야 합니다.<br>
암호는 아래에서 보이는 명령어로 꺼내서 복사 붙여넣기 합니다. 

{% highlight bash %}
$ sudo cat /var/lib/jenkins/secrets/initialAdminPassword
bf283■■■■■■■■■■■■■■■■■■■ee24
{% endhighlight %}

<img src="{{ page.asset_path }}jenkins-01.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

이후 설치하는 방식 2가지 버튼이 있는데, <br>
그냥 Install suggested plugins 를 누르면 대부분의 경우에서 다 잘됩니다.<br>
아래는 Install suggested plugins 를 선택했을때의 화면입니다.<br>
아래 plugin 모두 설치합니다. 

<img src="{{ page.asset_path }}jenkins-02.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

이후 Admin User를 생성합니다. 


<img src="{{ page.asset_path }}jenkins-03.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

모두 설치가 완료되면 다음과 같은 화면이 나옵니다.

<img src="{{ page.asset_path }}jenkins-04.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">



## 2.6 Plugins

1. Manage Jenkins -> Manage Plugins 
2. 다음을 설치 합니다. 
   1. **`Docker Pipeline`**
   2. **`Amazon ECR plugin`**
   

<img src="{{ page.asset_path }}jenkins-33.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">



# 3. Jenkins + Github Webhook

## 3.1 Creating GitHub Personal Access Token

Github의 우측상단에 자신의 프로필 사진을 누르고, `Setting -> Developer Settings -> Personal Access Tokens` 메뉴를 누릅니다.<br>
이름 넣어주고, scopes은 다음과 같이 선택합니다. 

 - repo
 - admin:repo_hook

<img src="{{ page.asset_path }}jenkins-13.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

생성하면 랜덤 문자열 같은게 생성 됩니다.  <br>
복사해서 Jenkins에 복사해줍니다. 


## 3.2 Jenkins Credential

Jenkins 초기화면에서 `Manage Jenkins -> Configure System` 메뉴에서 GitHub 계정을 설정하는 곳을 찾아서 설정해줍니다.<br>
이름은 적절하게 만들어주고, API URL은 https://api.github.com 으로 되어 있는데 그냥 default값 사용하고, <br> 
Manage hooks 체크박스 눌러줍니다. 

<img src="{{ page.asset_path }}jenkins-15.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

Credential 추가를 눌러서, Github에서 생성한 personal access token을  secret에다가 넣고, Github ID도 넣어줍니다. 

<img src="{{ page.asset_path }}jenkins-14.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">






## 3.3 Jenkins GitHub 설정

New Item 생성을 하며, Freestyle 을 선택합니다 .

<img src="{{ page.asset_path }}jenkins-10.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


Repository URL에 git 주소를 넣어줍니다. <br>
여기서는 https://github.com/AndersonJo/jenkins-tutorial.git 넣었습니다.<br>
추가적으로 branch 입력시 master 인지 main 인지도 확실하게 해줘야 합니다.

<img src="{{ page.asset_path }}jenkins-11.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

Credential 생성시에는 Login ID, Password 로 생성합니다. <br>

<img src="{{ page.asset_path }}jenkins-16.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

Build Triggers 에서는 `GitHub hook trigger for GITScm polling` 을 선택합니다.<br>
GitHub에 코드가 push되면 빌드를 하도록 설정하는 것 입니다. <br> 

push를 날리게 되면, GitHub에서 webhook 메세지를 Jenkins에 보내게 되며, <br> 
webhook 메세지를 받은 Jenkins는 이때부터 빌드를 진행하게 됩니다.


<img src="{{ page.asset_path }}jenkins-12.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


## 3.4 Webhook 설정

Gtihub Repository에 들어가서, 다음과 같이 설정 합니다.<br>
위치는 `Repository -> Settings -> Webhooks`


 - Jenkins 주소에 `/github-webhook/` 을 붙여줍니다. 
   - 예) `http://34.227.49.74:8080/github-webhook/`
   - 중요한점은 끝에 슬래쉬가 반드시 들어가야 합니다.

<img src="{{ page.asset_path }}jenkins-20.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

















# 4. Jenkins Pipeline

## 4.1 신규 아이템 생성

1. Jenkins -> Dashboard 에서 New Item 을 눌러서 새로운 아이템을 생성합니다. 
2. Pipeline 을 선택합니다. 

<img src="{{ page.asset_path }}jenkins-55.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


1. Pipeline 메뉴에서 `Pipeline script from SCM` 을 선택하고 Github 정보를 넣습니다. 
   1. Repository URL: http로 시작하는 github repository 주소. .git으로 끝남
   2. Credentials: 위에서 설정한 credential 추가
   3. Branch Specifier: master 인지 main 인지 잘 설정해야 함
   4. Script Path: jenkinsfile 의 파일 위치를 설정

<img src="{{ page.asset_path }}jenkins-56.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


## 4.2 Jenkinsfile Main Structure

jenkinsfile 은 그냥 텍스트 파일이고, 크게 3가지로 구성되어 있습니다.<br>
Groovy syntax 를 갖고 있으며, stage, 그리고 step 명령어 구조화하며,<br> 
credentials을 통해서 secret key등을 환경변수에서 가져올 수 있습니다. 

자세한 것은 [Documentation](https://www.jenkins.io/doc/book/pipeline/jenkinsfile/)을 참고 합니다.

{% highlight groovy %}
pipeline {
    agent any
    environment {
        AWS_ACCESS_KEY_ID     = credentials('jenkins-aws-secret-key-id')
        AWS_SECRET_ACCESS_KEY = credentials('jenkins-aws-secret-access-key')
    }
    stages {
        stage('Build') {
            steps {
                echo 'Building..'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}
{% endhighlight %}



## 4.2 Build Docker Image 


{% highlight groovy %}
pipeline {
    agent {
    }
    environment {
        AWS_ACCESS_KEY_ID     = credentials('jenkins-aws-secret-key-id')
        AWS_SECRET_ACCESS_KEY = credentials('jenkins-aws-secret-access-key')
    }
    stages {
        stage('Build Docker Image') {
            steps {
        } 
        }
        stage('Build') {
            steps {
                echo 'Building..'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}
{% endhighlight %}























