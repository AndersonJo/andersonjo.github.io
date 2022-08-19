---
layout: post
title:  "Kubernetes - Autoscaling (Cluster Scaler, Pod Scaler)"
date:   2022-08-06 01:00:00
categories: "kubernetes"
asset_path: /assets/images/
tags: ['hpa', 'ray']
---



# 1. Cluster AutoScaler

## 1.1 Brief Introduction

1. Cluster Autoscaler는 자동으로 nodes 의 갯수를 증가/감소를 합니다. 
2. Pod에 대한 autoscaling 은 아래쪽 HPA를 참조


## 1.2 IAM Policy

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
{% endhighlight %}
 
생성후 IAM -> Policy 를 보면 다음과 같은 Policy가 생성되어 있습니다. 


<img src="{{ page.asset_path }}kuberntes-autoscaler-policy.png" class="img-responsive img-rounded img-fluid border rounded" style="border:1px solid #aaa;">






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