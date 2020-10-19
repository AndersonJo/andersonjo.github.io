---
layout: post
title:  "Amazon Elastic Kubernetes Service (EKS) - Installation"
date:   2020-09-26 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['aws', 'mfa', 'otp', 'authentication', 'eks', 'kubeflow', 'dashboard', 'login', '로그인']
---

# 1. Introduction
 
Amazon EKS (Elastic Kubernetes Service)는 가장 손쉽게 Kubernetes를 바로 돌릴수 있는 서비스 입니다. <br>
EKS는 AWS안의 다른 서비스와 연결되어서 scalability 그리고 security 등을 제공하고 있습니다. 대표적으로 ..

 - **Amazon ECR**: Container images 
 - Elastic Load Balancing: Load distribution
 - IAM: Authentication
 - Amazon VPC: Isolation
 
EKS는 각 클러스터마다 각각의 control plane을 운영하며, control plane은 최소 2개의 API server nodes 를 가용하며, 3개의 etcd nodes를 가용합니다. 


# 2. Installation

## 2.1 AWS CLI

{% highlight bash %}
$ curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
$ unzip awscliv2.zip
$ sudo ./aws/install
{% endhighlight %}


{% highlight bash %}
$ aws configure
{% endhighlight %}

주요 Regions 
 - **us-east-2**: US EAST (Ohio)
 - **us-east-1**: US EAST (N. Virginia)
 - **us-west-1**: US West (N. California)
 - **us-west-2**: US West (Oregon)
 - **ap-east-1**: Asia Pacific (Hong Kong) 
 - **ap-northeast-1**: Asia Pacific (Tokyo)
 - **ap-northeast-2**: Asia Pacific (Seoul)
 - **ap-northeast-3**: Asia Pacific (Osaka-Local)
 - **ap-southeast-1**: Asia Pacific (Singapore)
 - **ap-southeast-2**: Asia Pacific (Sydney)

## 2.2 aws-iam-authenticator 

Amazon EKS는 IAM을 사용해서 EKS Cluter에 대한 authentication을 [aws-iam-authenticator](https://github.com/kubernetes-sigs/aws-iam-authenticator)를 통해서 제공 합니다. <br>
자세한 내용은 [aws-iam-authenticator 설치](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/install-aws-iam-authenticator.html) 문서를 참조 합니다. <br>


{% highlight bash %}
$ curl -o aws-iam-authenticator https://amazon-eks.s3.us-west-2.amazonaws.com/1.17.9/2020-08-04/bin/linux/amd64/aws-iam-authenticator
$ chmod +x ./aws-iam-authenticator
$ sudo cp ./aws-iam-authenticator /usr/local/bin/
{% endhighlight %}

## 2.3 AWS MFA 

만약 사내에서 Multi-Factor Authentication (MFA)를 사용중이라면, MFA인증이 필요합니다. <br>
MFA는 그냥 쉽게 말하면, 핸드폰에 Google OTP 앱으로 인증하는 것이라고 생각하면 됩니다. 

1. AWS Console 우측상단에 여러분의 아이디 -> My Security Credentials 를 누릅니다.
2. AWS IAM credentials -> Multi-factor authentication (MFA) -> Manage MFA device 버튼 눌러서 생성 <br> <span style="color:#555555">생성후 arn:aws:iam::123456788990:mfa/anderson.jo@google.com 이렇게 생긴 authentication code 생성됨</span> 
3. 구글 앱스토에서 Google OTP 앱을 다운로드 받고 MFA 설치 (이후 AWS Console 로그인은 모바일 OTP사용해서 인증) 
4. linux에서는 [aws-mfa](https://github.com/broamski/aws-mfa) 사용함

인증은 아래와 같이 하면 됩니다. 

{% highlight bash %}
$ aws-mfa --device arn:aws:iam::123456788990:mfa/anderson.jo@google.com
INFO - Validating credentials for profile: default 
INFO - Your credentials have expired, renewing.
Enter AWS MFA code for device [arn:aws:iam::123456788990:mfa/anderson.jo@google.com] (renewing for 43200 seconds):628066
INFO - Fetching Credentials - Profile: default, Duration: 43200
INFO - Success! Your credentials will expire in 43200 seconds at: 2020-09-26 22:02:35+00:00
{% endhighlight %}


 
## 2.4 EKSCTL

EKSCTL은 Amazon EKS의 공식 CLI툴입니다. 

* [EKSCTL Github](https://github.com/weaveworks/eksctl)

{% highlight bash %}
$ curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
$ sudo mv /tmp/eksctl /usr/local/bin
{% endhighlight %}

## 2.5 Kubectl

{% highlight bash %}
$ curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
$ # curl -o kubectl https://amazon-eks.s3.us-west-2.amazonaws.com/1.17.9/2020-08-04/bin/linux/amd64/kubectl
$ chmod +x ./kubectl
$ sudo mv ./kubectl /usr/local/bin
{% endhighlight %}

설치 확인은 버젼 체크로 할 수 있습니다.

{% highlight bash %}
$ kubectl version --short
Client Version: v1.19.3
Server Version: v1.17.9-eks-4c6976
{% endhighlight %}

.bashrc 에 다음의 넣습니다.

{% highlight bash %}
# kubectl autocomplete
source <(kubectl completion bash)

# kubectl shorthand 
alias k=kubectl
complete -F __start_kubectl k
{% endhighlight %}


# 3. Cluster 생성/로그인

## 3.1 IAM Settings 

1. [IAM Console](https://console.aws.amazon.com/iam/) 로 접속합니다.
2. Roles -> Create Role 
3. 서비스 리스트 중에서 EKS 선택 -> EKS 관련 Permission 선택
   - [`AmazonEKSClusterPolicy`](https://console.aws.amazon.com/iam/home#/policies/arn:aws:iam::aws:policy/AmazonEKSClusterPolicy%24jsonEditor) 는 반드시 선택
4. Role name은 eksRole, eksServiceRole 등등 적합한 단어로 생성

<img src="{{ page.asset_path }}eks-iam-eks-cluster-review.png" class="img-responsive img-rounded img-fluid" style="border: 2px solid #333333">

## 3.2 Create Cluster VPC with Cloud Formation 

 * Cluster생성시에 VPC subnets을 지정해야 하며, EKS는 최소 2 availability zones을 선택해야 합니다. 
 * 이때 1개는 public subnets, 다른 1개는 private subnets으로 사용하는게 좋습니다. 
    - public subnets: public load balancers를 사용해서 외부에서 받은 traffic을 내부 private subnet으로 보냅니다. 
    - private subnets: 실제 application을 실행시키는 pods이 존재하며, 해당 pods은 private subnets안에서 존재하는 nodes들로 이루어져 있습니다.

이때 VPC Network 설계가 좀 귀찮을수 있는데.. AWS에서는 Cloud Formation을 통해서 쉽게 해결이 가능합니다.<br>
아래 링크를 참조 합니다. 

 - [Two public subnets and two private subnets](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#vpc-public-private2)
 - [Trhee public subnets](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#vpc-public-only2)
 - [Three private subnets](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#vpc-private-only2)
 
기본적으로 Cloud Formation에서 VPC생성후 -> EKS Cluster를 생성할때 Cloud Formation으로 만들어 놓은 VPC ID, SecurityGroup, Subnet IDs 을 가져다 사용하는 것 입니다. <br>
본문에서는 Two public subnets 그리고 two private subnets 생성과 관련을 설명하겠습니다. 

1. 먼저 [CloudFormation](https://console.aws.amazon.com/cloudformation/)으로 접속 -> Create Stack -> With new resources(standard) 선택
2. 다음과 같이 선택하고 S3 URL을 넣습니다. <br>
   ```
   https://amazon-eks.s3.us-west-2.amazonaws.com/cloudformation/2020-08-12/amazon-eks-vpc-private-subnets.yaml
   ```
   <img src="{{ page.asset_path }}eks-cloudformation-select-s3-template.png" class="img-responsive img-rounded img-fluid" style="border: 2px solid #333333">
3. 나머지는 Subnets CIDR정도만 수정하고, 리뷰하고 생성 
4. 중요한점은 모두 생성되고나서, Outputs를 보고 SecurityGroups, SubnetIds, VpcId 을 따로 기록해둡니다. (Kubernetes생성시 사용) <br><br>
   <img src="{{ page.asset_path }}eks-cloudformation-outputs.png" class="img-responsive img-rounded img-fluid" style="border: 2px solid #333333">

* **SecurityGroups**: 추후 nodes를 cluster에 추가시킬때, 반드시 해당 SecurityGroup을 명시해야 합니다. 이를 통해서 EKS control plane과 nodes와 통신이 가능합니다. 
* **VpcId**: node group template 사용시 필요합니다. 
* **SubnetIds**: nodes를 cluster에 추가시에, 어떤 subnets으로 생성시킬지 명시해야 합니다.

## 3.3 Create Amazon EKS Cluster  

1. [Amazon EKS Console](https://console.aws.amazon.com/eks/home#/clusters) 을 열고 -> Create cluster 버튼을 누릅니다.
2. 적당히 이름넣고, Cluster Service Role은 이전에 만들었던 IAM Role을 넣습니다.<br><br>
   <img src="{{ page.asset_path }}eks-create-cluster-configuration.png" class="img-responsive img-rounded img-fluid" style="border: 2px solid #333333">
3. Networking은 Cloud Formation에서 생성한 VPC, Subnets, Security Groups을 선택합니다. 
4. 그외 로그 옵션 잘 선택하고 만들면 끝

## 3.4 Create Kubeconfig (Cluster 로그인) 

kubectl 명령어로 Cluster에 붙어서 명령을 내리려면 kubeconfig파일 설정이 되어 있어야 합니다. <br>
여기안에는 여러 정보가 있지만, 일단 Cluster에 접속해서 authentication을 하려면 Token을 얻어야 합니다.<br>
Token은 다음의 명령어로 얻을 수 있습니다. 

{% highlight bash %}
$ aws eks get-token --cluster-name AI-EKS-D | jq .status.token
{% endhighlight %}

물론 위의 명령어를 사용해서 [kubeconfig 파일은 manual](https://docs.aws.amazon.com/eks/latest/userguide/create-kubeconfig.html)로 만들어 줄 수도 있습니다.<br>
`~/.kube/admin.conf` 또는 `~/.kube/config` 파일을 자동으로 생성하기 위해서는 아래의 명령어를 사용합니다.

- `--role-arn <IAM role ARN>` : 해당 option을 추가해서 authentication을 사용할 수 있습니다.
jo
{% highlight bash %}
$ aws eks --region <us-west-2> update-kubeconfig --name <cluster_name>
{% endhighlight %}

kubectl명령어가 잘되는지 확인을 해 봅니다. 

{% highlight bash %}
$ kubectl get svc
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.100.0.1   <none>        443/TCP   74m
{% endhighlight %}

- Trouble Shooting: [Unauthorized or access denied Error (kubectl)](https://docs.aws.amazon.com/eks/latest/userguide/troubleshooting.html#unauthorized) 
 
## 3.5 Upgrade amazon-vpc-cni-k8s (to 1.7)

- [cni upgrade 하는 방법 AWS 문서](https://docs.aws.amazon.com/eks/latest/userguide/cni-upgrades.html)
- [Github amazon-vpc-cni-k8s](https://github.com/aws/amazon-vpc-cni-k8s) 

먼저 현재의 버젼을 확인합니다.

{% highlight bash %}
$ kubectl describe daemonset aws-node --namespace kube-system | grep Image | cut -d "/" -f 2
amazon-k8s-cni:v1.6.3-eksbuild.1
{% endhighlight %}

[버젼별 yaml](https://github.com/aws/amazon-vpc-cni-k8s/tree/master/config) 에 들어가서 최신 버젼을 확인합니다.<br>
이후 지역에 따라서 업그레이드 방법이 다릅니다. <br>
아래의 코드는 지역 상관없이 1.7을 설치하는 예제 입니다. 

{% highlight bash %}
$ curl -o aws-k8s-cni.yaml https://raw.githubusercontent.com/aws/amazon-vpc-cni-k8s/master/config/v1.7/aws-k8s-cni.yaml
$ sed -i -e 's/us-west-2/<region-code>/' aws-k8s-cni.yaml
$ kubectl apply -f aws-k8s-cni.yaml
{% endhighlight %}


## 3.7 EKS Cluster Endpoint Access Control 

Cluster를 생성시 Amazon EKS는 kubectl같은 툴로 커뮤니케이션 할 수 있도록 Kubernetes API Server의 endpoint를 생성합니다. <br>
기본값으로 해당 endpoint는 internet에 public으로 열려 있으며, 해당 접근은 IAM 또는 Kubernetes의 Role Based Access Control (RBAC)로 관리가 됩니다.

 - EKS -> Clusters -> 특정 Cluster 선택 -> Networking -> Manage Networking 버튼 선택<br><br>
   <img src="{{ page.asset_path }}eks-manage-networking.png" class="img-responsive img-rounded img-fluid" style="border: 2px solid #333333">


| Mode               | Public Access | Private Access | Description                 |
|:-------------------|:--------------|:---------------|:----------------------------|
| Public             | Enabled       | Disabled       | - Default 설정 <br> - Cluster VPC 내부에서 발생한 requests는 VPC를 나와서, Amazon Network로 통신 <br> - 외부에서 kubectl로 관리 가능       |
| Public and Private | Enabled       | Enabled        | - Public 도 enabled 하고, private도 enabled 한다는 뜻 <br> - Cluster VPC 내부 (node -> control plane) 의 경우 private VPC endpoint 사용 <br> - kubectl로 외부에서 관리 가능 | 
| Private            | Disabled      | Enabled        | - 외부에서 kubectl 접속 안됨 <br> - 모든 traffic은 cluster's VPC 내부에서만 허용됨   |

 - 그외 Source를 CIDR로 접속 제한 가능함 
 - Private only 로 했을경우 외부에서 접속이 안되는데.. 이 경우 다음과 같은 방법으로 Kubernetes API Server endpoint로 접속이 가능합니다. 
    - Conneted Network: [AWS transit gateway](https://docs.aws.amazon.com/vpc/latest/tgw/what-is-transit-gateway.html) 또는 [Connectivity](https://docs.aws.amazon.com/aws-technical-content/latest/aws-vpc-connectivity-options/introduction.html) 사용으로 해결
    - Amazon EC2 bastion host: EC2 instance를 Cluster's VPC안의 public subnet에 올려놓고, SSH로 login 한 다음 그 안에서 kubectl명령어를 사용하면 됨
    - AWS Cloud9 IDE 사용 
    
 
 
 
# 4. Node

두가지 방법으로 Compute를 생성할 수 있습니다. 

1. [Fargate - Linux](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#gs-console-fargate):  Fargate는 serverless compute engine으로서 서버 관리, 리소스 관리를 자동으로 해줍니다. 즉.. container의 사용한 만큼만 내면 됩니다.
2. [Managed nodes - Linux](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#gs-console-managed-nodes): 일반적인 방법

## 4.1 Managed Node 생성 

다만 문제는 Fargate의 경우 아직 제공되는 regions이 한정적이라, 이건 일단 패스하고 Managed Node 생성에 대해서 이야기 하겠습니다. <br>
먼저 Amazon EKS node role 생성이 필요합니다. 

1. [IAM Console](https://console.aws.amazon.com/iam/) 접속 -> Create Role 
2. Common use cases 에서 EC2 선택하고 바로 Next:Permission 버튼 클릭
3. Permissions 은 다음을 선택합니다. 
   - `AmazonEKSWorkerNodePolicy`
   - `AmazonEKS_CNI_Policy` 
   - `AmazonEC2ContainerRegistryReadOnly` 
6. Next: Tags 클릭
7. Role Name: `EKSNodeInstanceRole` 등의 유니크한 이름 생성 <br><br>
   <img src="{{ page.asset_path }}eks-managed-node-iam-role.png" class="img-responsive img-rounded img-fluid" style="border: 2px solid #333333">


그 다음으로 managed node group을 생성합니다. 

1. [Amazon EKS Console](https://console.aws.amazon.com/eks/home#/clusters) 열고 ->  Cluster 선택 -> Compute 탭 -> Add Node Group 선택   
2. Step3 Specify networking
   - subnets: Cloud Formation으로 생성한 subnets을 선택
   - Allow remote access to nodes: 생성후에 enable 시킬수 없고, SSH 접속이 안되니, 이건 거의 반드시 enable 시키고 시작함
   - SSH Key Pair 선택 (없으면 생성하면 됨)
3. 리뷰 예제는 다음과 같습니다. <br><br>
   <img src="{{ page.asset_path }}eks-node-creation-review.png" class="img-responsive img-rounded img-fluid" style="border: 2px solid #333333">

만약 GPU nodes의 경우 [Nvidia device plugin for Kubernetes](https://github.com/NVIDIA/k8s-device-plugin) 를 Cluster의 DaemonSet 으로 적용해야 합니다. <br>

{% highlight bash %}
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.6.0/nvidia-device-plugin.yml
{% endhighlight %}

잘 생성 됐는지 확인해 봅니다.

{% highlight bash %}
kubectl get nodes --watch
{% endhighlight %}



# 5. Test Installation

{% highlight bash %}
$ kubectl create deployment test-deployment --image=gcr.io/google-samples/kubernetes-bootcamp:v1
$ kubectl port-forward deployments/test-deployment 8080:8080
{% endhighlight %}

[http://localhost:8080](http://localhost:8080) 에서 확인합니다. <br>
이후 삭제 합니다.

{% highlight bash %}
$ kubectl delete deployment/test-deployment
{% endhighlight %}



# 6. Dashboard 

## 6.1 Install Dashboard 

먼저 Kubernetes Metrics Server를 설치해야 합니다. <br>
Metrics Server는 CPU, Memory같은 metrics 데이터를 수집하는 서버이며, EKS설치시 기본 서버에 설치되어 있지 않습니다. 

**Metrics Server deployment**는 그리고 확인은 다음의 명령어로 합니다.

{% highlight bash %}
$ kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.3.6/components.yaml
$ kubectl get deployment metrics-server -n kube-system
{% endhighlight %}

**Dashboard** 를 배포합니다.

{% highlight bash %}
$ kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0-beta8/aio/deploy/recommended.yaml
{% endhighlight %}

**eks-admin service account 그리고 cluster role binding** 을 만들어야 합니다. <br>
기본적으로 Kubernetes Dashboard user는 제한적인 권한만을 갖습니다. <br>
eks-admin service account 그리고 cluster role binding을 생성함으로서 Dashboard에 admin-level로 접속 할 수 있게 됩니다.<br> 

아래의 manifest는 `cluster-admin`(superuser) 권한을 갖고 있습니다. (그외 `amdin`, `edit`, `view` 등이 있음)<br>
자세한 내용은 [RBAC authorization](https://kubernetes.io/docs/admin/authorization/rbac/)을 참고합니다.

`vi eks-admin-service-account.yaml` 파일을 생성합니다. 

{% highlight yaml %}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: eks-admin
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1beta1
kind: ClusterRoleBinding
metadata:
  name: eks-admin
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: eks-admin
  namespace: kube-system
{% endhighlight %}

적용하고 확인합니다.

{% highlight bash %}
$ kubectl apply -f eks-admin-service-account.yaml
$ kubectl get serviceaccounts eks-admin -n kube-system
{% endhighlight %}

## 6.2 Access to Dashboard

**Dashboard 에 접속** 하기 위해서는 먼저 eks-admin service account에 대한 authentication token을 얻어야 합니다. 

{% highlight bash %}
$ kubectl -n kube-system describe secret $(kubectl -n kube-system get secret | grep eks-admin | awk '{print $1}')
$ kubectl proxy
{% endhighlight %}

[http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/#!/login](http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/#!/login) 로 접속합니다.

<img src="{{ page.asset_path }}eks-dashboard-token-auth.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


## 6.3 External Access to Dashboard 

**외부접속**을 하게 하려면 service "type"을 ClusterIP에서 NodePort로 변경하면 됩니다. 

{% highlight bash %}
$ kubectl -n kubernetes-dashboard edit service kubernetes-dashboard
{% endhighlight %}

{% highlight yaml %}
spec:
  clusterIP: 10.100.86.108
  ports:
  - port: 443
    protocol: TCP
    targetPort: 8443
  selector:
    k8s-app: kubernetes-dashboard
  sessionAffinity: None
  type: ClusterIP  ### 여기 ClusterIP를 NodePort 로 변경
{% endhighlight %}

서비스를 확인합니다. 

{% highlight bash %}
$ kubectl -n kubernetes-dashboard get services
NAME                        TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)         AGE
dashboard-metrics-scraper   ClusterIP   10.100.39.148   <none>        8000/TCP        173m
kubernetes-dashboard        NodePort    10.100.86.108   <none>        443:32080/TCP   173m
{% endhighlight %}

authentication token을 얻습니다.

{% highlight bash %}
$ kubectl -n kube-system describe secret $(kubectl -n kube-system get secret | grep eks-admin | awk '{print $1}')
{% endhighlight %}

이후 https://[master_node_ip]:[port] <br> 
로 접속을 할 수 있습니다.

# 7. Authentication

## 7.1 kubeconfig 파일 생성  

Cluster Authentication에 사용되는 Token 을 얻는 방법은 다음과 같이 합니다.

{% highlight bash %}
$ aws eks get-token --cluster-name AI-EKS-D | jq .status.token
{% endhighlight %}