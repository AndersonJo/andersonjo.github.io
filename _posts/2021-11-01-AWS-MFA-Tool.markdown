---
layout: post 
title:  "AWS Simple MFA"
date:   2021-11-01 01:00:00 
categories: "aws"
asset_path: /assets/images/ 
tags: []
---

<header>
    <img src="{{ page.asset_path }}kafka_background.jpeg" class="center img-responsive img-rounded img-fluid">
</header>

AWS MFA 관리 쉽게 하는 툴 입니다. 

# 1. Setting Up

## 1.1 Installation  

{% highlight bash %}
$ pip install aws-mfa
{% endhighlight %}


## 1.2 Credentials File Setup

{% highlight bash %}
$ vi ~/.aws/credentials
{% endhighlight %}

아래와 같이 설정을 합니다. 

{% highlight bash %}
[default-long-term]
aws_access_key_id = YOUR_LONGTERM_KEY_ID
aws_secret_access_key = YOUR_LONGTERM_ACCESS_KEY
aws_mfa_device = YOUR_MFA_ACCESS_KEY
{% endhighlight %}


## 1.3 MFA Login

My Security Credentials 에서 Multi-factor authentication (MFA) 에서 MFA Device ARN 을 복사합니다. 

{% highlight bash %}
$ aws-mfa 
{% endhighlight %}