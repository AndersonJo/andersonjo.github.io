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


<iframe width="560" height="315" src="https://www.youtube.com/embed/23vZrkQZ4Y8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



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

## 2.2 Docker Permission for Jenkins

Jenkins에서 docker를 실행할 수 있도록 해줘야 합니다.
jenkins 유저에게 docker 권한을 주는 것 입니다.

{% highlight bash %}
sudo usermod -a -G docker jenkins
sudo chmod 777 /var/run/docker.sock
{% endhighlight %}

## 2.3 Starting Jenkins

systemctl 로 시작하고, status를 통해서 상태를 확인합니다. 

{% highlight bash %}
$ sudo systemctl start jenkins
$ sudo systemctl status jenkins
{% endhighlight %}


## 2.4 OpenSSH

{% highlight bash %}
$ sudo apt-get install openssh-server
$ sudo systemctl enable ssh
$ sudo systemctl start ssh
{% endhighlight %}

설치이후 `ssh user@localhost` 같은 명령어로 테스트 해봅니다. 

## 2.5 Configure Firewall (Optional)

아래 코드는 ssh (port 22) 그리고 8080 포트를 여는 명령어 입니다. <br>
AWS는 Security Group에서 하면 됨으로 패스합니다. 

{% highlight bash %}
$ sudo apt-get install ufw
$ sudo ufw enable
$ sudo ufw allow ssh
$ sudo ufw allow 8080
{% endhighlight %}

## 2.6 Setting Up Jenkins

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



## 2.7 Plugins

1. Manage Jenkins -> Manage Plugins 
2. 다음을 설치 합니다. 
   1. **`CloudBees AWS Credentials Plugin`** 
   2. **`Docker Pipeline`** 
   3. **`Amazon ECR plugin`** 
   4. **`Kubernetes CLI`** (작동 안함. 하지만 일단 설치)
   

<img src="{{ page.asset_path }}jenkins-33.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">



## 2.8 Set AWS Credentials 

1. 가장 쉽게 credentials을 알아내는 방법은.. `cat ~/.aws/credentials` 명령어로 이미 설정되어 있는 credentials 을 꺼내는 것입니다.
2. 또는 IAM -> Users -> Security credentials -> Create Access Key 를 생성할수 있습니다. 

그래서 필요한건 `ACCESS KEY` 그리고 `SECRET KEY` 두개 입니다. 

Jenkins에서 설정은 다음과 같이 합니다.

1. Dashboard -> Manage Jenkins -> Manage Credentials
2. 아래와 같은 화면에서 Stores -> `(global)` 누릅니다. -> 누르면 왼쪽에 `Add Credentials` 를 누릅니다. 

<img src="{{ page.asset_path }}jenkins-58.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


생성은 다음과 같이 참고해서 만듭니다.<br>
ID (jenkins-aws-anderson-credentials) 는 Jenkins Pipeline에서 다시 사용 됩니다. 

<img src="{{ page.asset_path }}jenkins-57.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">












# 3. Jenkins + Github Webhook

## 3.1 Webhook 설정

Gtihub Repository에 들어가서, 다음과 같이 설정 합니다.<br>
위치는 `Repository -> Settings -> Webhooks`


 - Jenkins 주소에 `/github-webhook/` 을 붙여줍니다. 
   - 예) `http://34.227.49.74:8080/github-webhook/`
   - 중요한점은 끝에 슬래쉬가 반드시 들어가야 합니다.

<img src="{{ page.asset_path }}jenkins-20.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


## ~~3.2 Creating GitHub Personal Access Token~~

* Jenkins 버그로 인해서 작동을 안하고, 그냥 아래의 Github ID, Password 방식으로 해야 작동함

Github의 우측상단에 자신의 프로필 사진을 누르고, `Setting -> Developer Settings -> Personal Access Tokens` 메뉴를 누릅니다.<br>
이름 넣어주고, scopes은 다음과 같이 선택합니다. 

 - repo
 - admin:repo_hook

<img src="{{ page.asset_path }}jenkins-13.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

생성하면 랜덤 문자열 같은게 생성 됩니다.  <br>
복사해서 Jenkins에 복사해줍니다. 



## 3.2 ~~Jenkins Credential~~

Jenkins 초기화면에서 `Manage Jenkins -> Configure System` 메뉴에서 GitHub 계정을 설정하는 곳을 찾아서 설정해줍니다.<br>
이름은 적절하게 만들어주고, API URL은 https://api.github.com 으로 되어 있는데 그냥 default값 사용하고, <_br> 
Manage hooks 체크박스 눌러줍니다. 

<img src="{{ page.asset_path }}jenkins-15.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

Credential 추가를 눌러서, Github에서 생성한 personal access token을  secret에다가 넣고, Github ID도 넣어줍니다. 

<img src="{{ page.asset_path }}jenkins-14.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">_









# 4. Jenkins Pipeline and ECR

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



Repository URL에 git 주소를 넣어줍니다. <br>
여기서는 https://github.com/AndersonJo/jenkins-tutorial.git 넣었습니다.<br>
추가적으로 branch 입력시 master 인지 main 인지도 확실하게 해줘야 합니다.

<img src="{{ page.asset_path }}jenkins-11.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

<span style="color:red">**Credential 생성시에는 Login ID, Password 로 생성합니다.**<br>
(Secret Text로 하면 Jenkins 버그로 인해서 dropbox에서 보이지 않는 일이 발생합니다. <br>
반드시 Github UserID 그리고 Password로 해야 합니다. <br>
또한 유저ID는 이메일이 아니라 Github UserID로 해야 합니다.<br>
예를 들어 제 Github UserID는 **AndersonJo** 입니다. )
</span> <br>
관련된 버그는 [링크](https://github.com/jenkinsci/ghprb-plugin/issues/534) 에서 확인 가능합니다. 

<img src="{{ page.asset_path }}jenkins-16.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

Build Triggers 에서는 `GitHub hook trigger for GITScm polling` 을 선택합니다.<br>
GitHub에 코드가 push되면 빌드를 하도록 설정하는 것 입니다. <br> 

push를 날리게 되면, GitHub에서 webhook 메세지를 Jenkins에 보내게 되며, <br> 
webhook 메세지를 받은 Jenkins는 이때부터 빌드를 진행하게 됩니다.


<img src="{{ page.asset_path }}jenkins-12.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">




## 4.2 Jenkinsfile & Docker Build & Push to ECR

jenkinsfile 은 그냥 텍스트 파일이고, 크게 3가지로 구성되어 있습니다.<br>
Groovy syntax 를 갖고 있으며, stage, 그리고 step 명령어 구조화하며,<br> 
credentials을 통해서 secret key등을 환경변수에서 가져올 수 있습니다. 

자세한 것은 [Documentation](https://www.jenkins.io/doc/book/pipeline/jenkinsfile/)을 참고 합니다.

Jenkinsfile 파일은 해당 github repository에 넣으면 됩니다. (Jenkins 어딘가 X)
Git Push를 하면, webhook으로 Jenkins에서 전달받게 되고, 해당 git repository를 checkout하게 됩니다.<br>
이후 Jenkins는 모두 다 다운받은후 -> 아래 빌드를 순차적으로 하게 됩니다. 

{% highlight groovy %}
node {
    stage('Clone Repository'){
        checkout scm
    }

    stage('Build to ECR'){

    }
    stage('Kubernetes'){
        
    }
}
{% endhighlight %}

아래 그림은 성공적으로 진행됐을 경우의 Jenkins 화면입니다. 

<img src="{{ page.asset_path }}jenkins-59.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">










# 5. Kubernetes and Jenkins

## 5.1 Jenkins Kubernetes CLI Plugin


Jenkins에서는 jenkins 리눅스 유저를 사용해고, 여기서 EKS 로 로그인이 필요합니다. <br>
Jenkins 플러그인중에 [Kubernetes CLI Plugin](https://plugins.jenkins.io/kubernetes-cli/) 에서 이런 기능을 제공합니다.<br>
중요한건 사용을 하기 위해서는 Jenkins에서 EKS 로그인에 필요한 credentials을 만들어야 합니다. 



{% highlight bash %}
{% raw %}
$ kubectl create serviceaccount jenkins-deployer
$ kubectl create clusterrolebinding jenkins-deployer-role --clusterrole==cluster-admin --serviceaccount=default:jenkins-deployer
$ kubectl get secrets jenkins-deployer-token-vgsfj -o go-template --template '{{index .data "token"}}' | base64 -d
eyJhbGciOiJSUzI1NiIsImtpZCI6IlJNQjV6QlNLT<생략 암호가 나오고 복사함!>
{% endraw %}
{% endhighlight %}

다음 명령어로 새로 만들어진  credentials을 복사합니다. 



<img src="{{ page.asset_path }}jenkins-72.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


이후 해당 Credentials 을 Jenkins 에 등록을 합니다.

- Manage Jenkins -> Manage Credentials -> 아무거나 (global) 선택 -> Add Credentials 선택 
- Secret: 복사한 token을 붙여넣습니다.
- ID: kubectl-deploy-credentials 

<img src="{{ page.asset_path }}jenkins-73.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


## 5.2 다필요없고 제일 쉬운 방법

문제가 되는 부분이 jenkins 유저로 돌아갈때 kubectl 이 안되는 문제가 있습니다.. <br>
정확하게는.. Jenkins Kubernetes CLI Plugin에 문제가 있는듯 하고.. <br>
아직까지 이걸 되게 만드려면 매우 어려움이 있습니다. <br>
따라서 그냥 제일 쉽게 하는 방법은 그냥 jenkins 유저로 접속해서 미리 인증 받아놓는 것이다. 

먼저 기존 ACCESS KEY 그리고 SECRET KEY를 파악한다. 

{% highlight bash %}
$ cat ~/.aws/credentials 
[default]
aws_access_key_id = AKIA4A27BB2QXSFEVPOE
aws_secret_access_key = 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
{% endhighlight %}

이후 jenkins 유저로 접 속해서 인증한다

{% highlight bash %}
$ sudo su jenkins
$ aws configure
AWS Access Key ID [****************VPOE]: 
AWS Secret Access Key [****************5Vq/]: 
Default region name [us-east-1]: 
Default output format [json]:
{% endhighlight %}

~~`aws eks --region <us-west-2> update-kubeconfig --name <cluster_name>` 명령어로 authentication 한다~~

## 5.3 Kubernetes Plugin

Jenkins에서 Kubernetes plugin을 설치합니다. 

 - **Kubernetes Plugin** 
 - **Kubernetes CLI**

<img src="{{ page.asset_path }}jenkins-70.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

<img src="{{ page.asset_path }}jenkins-71.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">


## 5.4 Pipeline

 - EKS_API: EKS Cluster -> 당신의 Cluster -> API server endpoint 
   - 예) `https://6918042C2B9B60669CFCE2B59402AF83.gr7.ap-northeast-2.eks.amazonaws.com`
 - EKS_CLUSTER_NAME: EKS Cluster의 이름
   - 예) `EKS-AI-Cluster`
 - EKS_NAMESPACE: 적용하려는 kubernetes의 namespace
 - EKS_JENKINS_CREDENTIAL_ID
   - `kubectl get secrets`
   - `kubectl describe secret default-token-k7bst`
 - ECR_REGION: ECR Region을 적으면 됨
 - ECR_PATH: ECR로 가서 Repository의 URI을 가져오되 `/repository-name` 은 제거한다

{% highlight groovy %}
REGION = 'ap-northeast-2'
EKS_API = 'https://6918042C2B9B60669CFCE2B59402AF83.gr7.ap-northeast-2.eks.amazonaws.com'
EKS_CLUSTER_NAME='EKS-AI-Cluster'
EKS_NAMESPACE='default'
EKS_JENKINS_CREDENTIAL_ID='kubectl-deploy-credentials'
ECR_PATH = '998902534284.dkr.ecr.ap-northeast-2.amazonaws.com'
ECR_IMAGE = 'test-repository'
AWS_CREDENTIAL_ID = 'aws-credentials'

node {
    stage('Clone Repository'){
        checkout scm
    }
    stage('Docker Build'){
        // Docker Build
        docker.withRegistry("https://${ECR_PATH}", "ecr:${REGION}:${AWS_CREDENTIAL_ID}"){
            image = docker.build("${ECR_PATH}/${ECR_IMAGE}", "--network=host --no-cache .")
        }
    }
    stage('Push to ECR'){
        docker.withRegistry("https://${ECR_PATH}", "ecr:${REGION}:${AWS_CREDENTIAL_ID}"){
            image.push("v${env.BUILD_NUMBER}")
        }
    }
    stage('CleanUp Images'){
        sh"""
        docker rmi ${ECR_PATH}/${ECR_IMAGE}:v$BUILD_NUMBER
        docker rmi ${ECR_PATH}/${ECR_IMAGE}:latest
        """
    }
    stage('Deploy to K8S'){
        withKubeConfig([credentialsId: "kubectl-deploy-credentials",
                        serverUrl: "${EKS_API}",
                        clusterName: "${EKS_CLUSTER_NAME}"]){
            sh "sed 's/IMAGE_VERSION/${env.BUILD_ID}/g' service.yaml > output.yaml"
            sh "aws eks --region ${REGION} update-kubeconfig --name ${EKS_CLUSTER_NAME}"
            sh "kubectl apply -f output.yaml"
            sh "rm output.yaml"
        }
    }
}
{% endhighlight %}

## 5.5 최종 확인

{% highlight bash %}
$ kubectl get svc
NAME                         TYPE           CLUSTER-IP       EXTERNAL-IP                                                               PORT(S)        AGE
kubernetes                   ClusterIP      10.100.0.1       <none>                                                                    443/TCP        17h
nginx-service-loadbalancer   LoadBalancer   10.100.160.221   a171eb183b72a45d8a93a11eab57c1bb-1295299660.us-east-1.elb.amazonaws.com   80:31063/TCP   13h
{% endhighlight %}

external IP로 들어간후.. nginx text 내용 변경하면.. 변경된 내용 배포됨

<img src="{{ page.asset_path }}jenkins-74.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">

<img src="{{ page.asset_path }}jenkins-75.png" class="center img-responsive img-rounded img-fluid" style="border:1px solid #aaa; max-width:800px;">



# 6. Code

## 6.1 Flask App 

{% highlight python %}
import os
from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    name = request.args.get('name', 'World')
    return 'Hello {}!\n'.format(name)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
{% endhighlight %}

## 6.2 Dcokerfile

{% highlight bash %}
FROM python:3.8
MAINTAINER Anderson Jo
USER root
ENV PYTHONUNBUFFERED=True
ENV APP_HOME=/app
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME
COPY . ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt
CMD exec gunicorn --bind 0.0.0.0:80 --workers 4 --threads 8 app:app
{% endhighlight %}

## 6.3 Jenkinsfile

{% highlight bash %}
REGION = 'ap-northeast-2'
EKS_API = 'https://6918042C2B9B60669CFCE2B59402AF83.gr7.ap-northeast-2.eks.amazonaws.com'
EKS_CLUSTER_NAME='test-cluster'
EKS_NAMESPACE='default'
EKS_JENKINS_CREDENTIAL_ID='kubectl-deploy-credentials'
ECR_PATH = '998902534284.dkr.ecr.ap-northeast-2.amazonaws.com'
ECR_IMAGE = 'test-repository'
AWS_CREDENTIAL_ID = 'aws-credentials'

node {
    stage('Clone Repository'){
        checkout scm
    }
    stage('Docker Build'){
        // Docker Build
        docker.withRegistry("https://${ECR_PATH}", "ecr:${REGION}:${AWS_CREDENTIAL_ID}"){
            image = docker.build("${ECR_PATH}/${ECR_IMAGE}", "--network=host --no-cache .")
        }
    }
    stage('Push to ECR'){
        docker.withRegistry("https://${ECR_PATH}", "ecr:${REGION}:${AWS_CREDENTIAL_ID}"){
            image.push("v${env.BUILD_NUMBER}")
        }
    }
    stage('CleanUp Images'){
        sh"""
        docker rmi ${ECR_PATH}/${ECR_IMAGE}:v$BUILD_NUMBER
        docker rmi ${ECR_PATH}/${ECR_IMAGE}:latest
        """
    }
    stage('Deploy to K8S'){
        withKubeConfig([credentialsId: "kubectl-deploy-credentials",
                        serverUrl: "${EKS_API}",
                        clusterName: "${EKS_CLUSTER_NAME}"]){
            sh "sed 's/IMAGE_VERSION/${env.BUILD_ID}/g' service.yaml > output.yaml"
            sh "aws eks --region ${REGION} update-kubeconfig --name ${EKS_CLUSTER_NAME}"
            sh "kubectl apply -f output.yaml"
            sh "rm output.yaml"
        }
    }
}
{% endhighlight %}


## 6.3 requirements.txt

{% highlight bash %}
# Server
Flask>=2.0.2
Flask-Cors>=3.0.10
gunicorn>=20.1.0
{% endhighlight %}


## 6.4 service.yaml

{% highlight yaml %}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-flask
  labels:
    app: hello-flask
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hello-flask
  template:
    metadata:
      labels:
        app: hello-flask
    spec:
      containers:
      - name: flask-app
        image: 998902534284.dkr.ecr.ap-northeast-2.amazonaws.com/test-repository:IMAGE_VERSION
        imagePullPolicy: Always
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: hello-flask-service
spec:
  type: LoadBalancer
  selector:
    app: hello-flask
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
{% endhighlight %}













