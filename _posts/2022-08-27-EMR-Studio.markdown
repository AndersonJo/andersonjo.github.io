---
layout: post
title: "EMR Studio"
date:  2022-08-27 01:00:00
categories: "spark"
asset_path: /assets/images/
tags: ['hadoop', 'spark']
---





# 1. Preparation

## 1.1 Iam Identity Center

EMR은 `IAM Authentication Mode` 그리고 `IAM Identity Center Authentication Mode` 두가지로 운영이 가능합니다.<br>
쉽게 이야기 하면.. 

- IAM Authentication Mode: 외부 인증앱 사용
- IAM Identity Center Authentication Mode: AWS 에서 제공하는 IAM Identity Center 사용해서 인증 가능. <- 이게 쉬움

그래서 본문에서는 후자를 사용할 것이고, 아래와 같이 생성합니다.<br>
Iam Identity Center -> Users -> Create

<img src="{{ page.asset_path }}emr-iam-ideneity-center-02.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

나머지는 전부다 Next 버튼 누르고 완료하면 됩니다.<br>
Group은 만들어도 되고 안해도 되고.. 맘대로..<br>
이후 **이메일 확인해서 인증**해야 합니다. 

**중요한 부분은 이것만 만들어서 끝이 아니고, 아래에 EMR Studio를 생성후, 여기서 만든 유저를 다시 추가 해야 합니다.**



## 1.2 EMR Studio User Role

원래는 개개인에 대한 권한을 부여해야 되지만, <br>
단순화하기 위해서 EMR Studio 를 사용하는 모든 유저에게<br> 
아래와 같은 동일한 권한을 부여하도록 Role 을 생성합니다. 

IAM -> Roles -> Create Role

 - Trusted entity type: AWS service
 - Use case: `EMR` <- 요거 찾아서 선택

<img src="{{ page.asset_path }}emr-studio-20.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

나머지는 다 넘기고 아래와 같이 몇가지만 설정합니다. 

- Role Name: `emr-studio-user-role`

생성이후에 emr-studio-role을 누르고 `Add Permissions` -> `Attach Policies` 를 눌러서 더 추가 합니다.<br>
최종적으로 4개가 추가되야 합니다. 

1. AmazonElasticMapReduceRole
2. AmazonS3FullAccess
3. AmazonEMRFullAccessPolicy_v2
4. AmazonEC2FullAccess

<img src="{{ page.asset_path }}emr-studio-23.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


## 1.3 EMR Studio Service Role 

위와 동일하게 만들되 이름은 `emr-studio-service-role` 로 만듭니다. <br>
최종 권한은 다음이 있으면 됩니다.  

1. AmazonElasticMapReduceEditorsRole
2. AmazonS3FullAccess
3. AmazonEC2FullAccess














# 2. EMR Studio

## 2.1 Create EMR Studio

먼저 EMR Studio 에서 Create Studio 버튼을 눌러서 다음과 같이 생성합니다.

 - Studio name: `emr-studio`
 - VPC: EMR과 동일한 VPC 선택 
 - subnets: EMR과 동일한 subnets 선택
 - Security and access: Default security group
 - Authentication: `AWS Identity and Access Management (IAM)`
 - Workspace storage: 저장할 S3 위치 지정

<img src="{{ page.asset_path }}emr-studio-01.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

<img src="{{ page.asset_path }}emr-studio-02.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

꽤 중요한 부분인데, `AWS Single Sign-On (SSO)` 를 선택하고, 그 아래 옵션은 다음과 같이 설정합니다. 

 - Service role: `emr-studio-user-role`
 - User role: `emr-studio-user-role`


<img src="{{ page.asset_path }}emr-studio-creation-authenticaion-role.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">


EMR Studio 생성이 완료되고 해당 studio를 누르면 detail page 화면에서 링크를 찾을 수 있습니다. <br>
EMR -> EMR Studio (emr-studio 방금 만든것) -> URL 클릭하면 studio 화면으로 이동


## 2.2 Add a user to EMR Studio

Iam Identity Center 에서 만든 Anderson 유저를 추가해줘야 합니다. 

<img src="{{ page.asset_path }}emr-studio-add-user.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

<img src="{{ page.asset_path }}emr-studio-add-user-02.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">

이후 이름으로 인증 (Anderson 으로 인증.. 이메일이 아니라..) 하고 나면 Workspace Dashboard 가 보입니다.





## 2.3 EMR Workspace

EMR Studio 안으로 들어오면 보이는 화면입니다. <br>
여기서 workspace를 생성할 수 있습니다. <br>
Create Workspace 버튼을 누릅니다. 

아래와 같이 workspace 를 생성하되, 반드시 advanced configuration 을 열고 `Attach Workspace to an EMR cluster` 를 선택합니다. 

<img src="{{ page.asset_path }}emr-workspace-05.png" class="img-responsive img-rounded img-fluid center" style="border: 2px solid #333333">




