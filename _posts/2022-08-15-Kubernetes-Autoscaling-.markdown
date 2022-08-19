---
layout: post
title:  "Kubernetes - Cluster Scaler"
date:   2022-08-15 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['hpa', 'ray', 'pod', 'scaling']
---



# 1. Cluster AutoScaler

## 1.1 Brief Introduction

1. Cluster Autoscaler는 자동으로 nodes 의 갯수를 증가/감소를 합니다. 
2. Pod에 대한 autoscaling 은 아래쪽 HPA를 참조


## 1.2 Autoscaling Group 

eksctl 로 EKS cluster를 만들었다면 이미 만들어져 있고, tag 만 정해주면 되지만, console 에서 만들었을경우 다음을 실행하여
AWS EC2 에서 Autoscaling Group 을 만들어줍니다. 

 - `EKS_CLUSTER_NAME`: 당신이 만들 EKS Cluster 이름
 - `EKS_NODEGROUP_NAME`: AutoScaling Group 적용하려는 Node Group의 이름

{% highlight bash %}
$ export EKS_CLUSTER_NAME=<My EKS Cluster Name>
$ export EKS_NODEGROUP_NAME=<My EKS Node Group for AutoScaling>
$ export ASG_NAME=$(aws eks describe-nodegroup --cluster-name ${EKS_CLUSTER_NAME} --nodegroup-name ${EKS_NODEGROUP_NAME} --query "nodegroup.resources.autoScalingGroups" --output text)
{% endhighlight %}

Auto Scaling Group 생성/업데이트를 합니다.

{% highlight bash %}
# 생성 
$ aws autoscaling \
    create-auto-scaling-group \
    --auto-scaling-group-name ${ASG_NAME} \
    --min-size 1 \
    --desired-capacity 1 \
    --max-size 5

# 업데이트
$ aws autoscaling \
    update-auto-scaling-group \
    --auto-scaling-group-name ${ASG_NAME} \
    --min-size 1 \
    --desired-capacity 1 \
    --max-size 5
{% endhighlight %}

## 1.3 IAM OIDC provider

1. 만들어놓은 EKS Cluster 클릭하면 `OpenID Connect provider URL` 있고, 이것을 복사합니다.<br>
   <img src="{{ page.asset_path }}kuberntes-autoscaler-openid.png" class="img-responsive img-rounded img-fluid border rounded center" style="border:1px solid #aaa;">
2. IAM -> Identity Provider (자격 증명 공급자) -> Add Provider (공급자 추가) 선택 
3. 다음 옵션으로 생성 
   - OpenID Connect 선택
   - Provider: 복사한 EKS Cluster의 OpenID Connect Provider URL 붙여넣기 -> Get Thumbprint
   - Audience (대상): `sts.amazonaws.com` 
   <img src="{{ page.asset_path }}kuberntes-autoscaler-openid2.png" class="img-responsive img-rounded img-fluid border rounded center" style="border:1px solid #aaa;">



## 1.4 IAM Policy

먼저 아래 코드를 실행시켜서 cluster-autoscaler-policy.json 파일을 생성합니다.<br>
이때 my-cluster 로 되어 있는 부분은 변경이 필요합니다. <br>
즉 `"aws:ResourceTag/k8s.io/cluster-autoscaler/my-cluster": "owned"` <- my-cluster 수정 필요.

{% highlight json %}
cat <<EOF > cluster-autoscaler-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "autoscaling:SetDesiredCapacity",
                "autoscaling:TerminateInstanceInAutoScalingGroup"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "aws:ResourceTag/k8s.io/cluster-autoscaler/my-cluster": "owned"
                }
            }
        },
        {
            "Sid": "VisualEditor1",
            "Effect": "Allow",
            "Action": [
                "autoscaling:DescribeAutoScalingInstances",
                "autoscaling:DescribeAutoScalingGroups",
                "ec2:DescribeLaunchTemplateVersions",
                "autoscaling:DescribeTags",
                "autoscaling:DescribeLaunchConfigurations"
            ],
            "Resource": "*"
        }
    ]
}
EOF
{% endhighlight %}


다음의 명령어로 IAM Policy 를 생성합니다. 

{% highlight bash %}
$ aws iam create-policy \
    --policy-name AmazonEKSClusterAutoscalerPolicy \
    --policy-document file://cluster-autoscaler-policy.json

# 아래는 결과값 
<생략>
    "Policy": {
        "PolicyName": "AmazonEKSClusterAutoscalerPolicy",
        "PolicyId": "ABCDEFGHIJKLMNOPQRST",
        "Arn": "arn:aws:iam::123456789012:policy/AmazonEKSClusterAutoscalerPolicy",
<생략>
{% endhighlight %}
 
위의 결과값중에서 ARN 값을 다시 사용하니 다른 곳에 적어둡니다. <br>
또한 생성후 IAM -> Policy 를 보면 다음과 같은 Policy가 생성되어 있는 것을 확인 할 수 있습니다.<br>
해당 정책을 클릭해서 보면, 여기에서도 ARN을 확인 할 수 있습니다. 

<img src="{{ page.asset_path }}kuberntes-autoscaler-policy.png" class="img-responsive img-rounded img-fluid border rounded center" style="border:1px solid #aaa;">



## 1.5 IAM Role 

1. AWS Console -> IAM -> Roles 이동 -> Create Role 클릭 (역활 만들기 버튼)
2. 아래 옵션으로 생성
   - Trusted Entity Type: Web Identity (웹 자격 증명)
   - Provider: 이전에 만들었던 openID 선택
   - Audience: `sts.amazonaws.com`
   <img src="{{ page.asset_path }}kuberntes-autoscaler-role1.png" class="img-responsive img-rounded img-fluid border rounded center" style="border:1px solid #aaa;">
3. 다음 옵션으로 설정
   - Role Name: `AmazonEKSClusterAutoscalerRole`
   - Description: `Amazon EKS - Cluster autoscaler role`

생성!

## 1.6 Deploy the Cluster Autoscaler

{% highlight bash %}
# 일단 Cluster Autoscaler YAML 파일 다운로드
$ curl -o cluster-autoscaler-autodiscover.yaml https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml
{% endhighlight %}

`cluster-autoscaler-autodiscover.yaml` 파일을 열고, 다음을 수정

- `<YOUR CLUSTER NAME>` 부분을 당신의 cluster name 으로 수정
- `cpu` 그리고 `memory` 부분을 환경에 맞게 수정

이후 배포 합니다. 

{% highlight bash %}
$ kubectl apply -f cluster-autoscaler-autodiscover.yaml 
{% endhighlight %}


## 1.7 Service Account & 그외 설정

아래 코드를 수정후 실행합니다. 

- `<ACCOUNT_ID>` 그리고 `<AmazonEKSClusterAutoscalerRole>` : `AmazonEKSClusterAutoscalerRole` 이름으로 Role을 위에서 만들었는데, 해당 Role의 ARN 을 복사해서 넣으면 됩니다.

{% highlight bash %}
$ kubectl annotate serviceaccount cluster-autoscaler \
  -n kube-system \
  eks.amazonaws.com/role-arn=arn:aws:iam::<ACCOUNT_ID>:role/<AmazonEKSClusterAutoscalerRole> 
{% endhighlight %}

annotation을 추가합니다.  (그냥 실행하면 됨)

{% highlight bash %}
$ kubectl patch deployment cluster-autoscaler \
  -n kube-system \
  -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"}}}}}'
{% endhighlight %}


이후 edit 으로 Cluster Autoscaler deployment 를 수정합니다. 

{% highlight bash %}
$ kubectl -n kube-system edit deployment.apps/cluster-autoscaler
{% endhighlight %}

`cluster-autoscaler` contianer 부분의 command 부분에 다음을 추가 합니다. 
- `--balance-similar-node-groups` : ensures that there is enough available compute across all availability zones
- `--skip-nodes-with-system-pods=false` : ensures that there are no problems with scaling to zero

수정후 아래와 같이 보일 것입니다.

{% highlight yaml %}
     app: cluster-autoscaler
 spec:
   containers:
   - command:
     - ./cluster-autoscaler
     - --v=4
     - --stderrthreshold=info
     - --cloud-provider=aws
     - --skip-nodes-with-local-storage=false
     - --expander=least-waste
     - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/my-cluster
     - --balance-similar-node-groups
     - --skip-nodes-with-system-pods=false
{% endhighlight %}


이후 [Cluster Autoscaler releases](https://github.com/kubernetes/autoscaler/releases) 에 들어가서 만들어놓은 Kubernetes 버젼과 동일한 Auto Scaler 버젼을 다운로드 받습니다.<br>
예를 들어 당신의 Kubernetes Cluster 버젼이 1.22 라면, 마찬가지로 Auto Scaler 버젼도 1.22 로 시작하는 Images 를 찾아서 <br>
아래 코드의 `cluster-autoscaler=` 요 부분을 대체 합니다. 

{% highlight bash %}
$ kubectl set image deployment cluster-autoscaler \
  -n kube-system \
  cluster-autoscaler=k8s.gcr.io/autoscaling/cluster-autoscaler:v1.22.3
{% endhighlight %}


## 1.8 AutoScaler Logs 

log는 다음의 명령어로 확인 가능합니다. 

{% highlight bash %}
$ kubectl -n kube-system logs -f deployment.apps/cluster-autoscaler
{% endhighlight %}















# 2. Horizontal Pod Autoscaler

## 2.1 Simple Introduction

1. Pod 에 대한 autoscaling과 관련된 내용입니다. 
2. 만약 node

작동방식은 다음과 같습니다. 

1. HPA 지속적으로(continuous) 도는 프로세스가 아니라, 간헐적으로(intermittenly) 돌아가는 프로세스이며, <br> 
   interval 은 `--horizontal-pod-autoscaler-sync-period` 로 설정할 수 있다 (기본값은 15초)
2. `Controller Manager` 라는 녀석이 autoscaling 을 컨트롤하며, 주기적으로 pod 을 `.spec.selector` 레이블을 `scaleTargetRef` 로 검색해서 찾는다. <br>
   이후 HPA 설정한 조건들과 현재 메트릭값을 비교하게 된다
3. 만약 CPU기준으로 HPA 가 설정했지만, 해당 pod안의 containers 들에서 resource request 가 정의되어 있지 않다면, autoscaler는 그냥 무시를 하게 된다.<br> 
   따라서 HPA가 정상 작동하기 위해서는 CPU, memory 등의 설정을 꼭 해주어야 한다.



## 2.2 Dependencies 

1. [metric-server](https://github.com/kubernetes-sigs/metrics-server)
정