---
layout: post
title:  "AWS API Gateway & Lambda & Dynamo"
date:   2016-03-08 01:00:00
categories: "aws"
static: /assets/posts/AWS/
tags: ['API Gateway', 'Lambda']

---

# Authorizer with API Gateway & Lambda 

### Custom Authorizer Architecture

<img src="{{ page.static }}custom-auth-workflow.png" class="img-responsive img-rounded">

OAuth, SAML 등 다양한 authentication 방법들을 API Gateway의 custom authorization을 통해서 컨트롤 할 수 있습니다.
Client가 request를 API Gateway로 authorization token을 헤더로 포함해서 보내면, 해당 Request를 Lambda로 보내고, 
Lambda는 다시 IAM Policies를 리턴시켜서 보냅니다. Policy가 유효하지 않거나, Denied가 되면, 해당 API에대한 call은 실패하게 됩니다.
Valid Policy를 보낸다면, API Gateway는 returned policy를 캐쉬시키고, 동일한 토큰을 갖고서 요청하는 모든 requests를 
미리 설정된 TTL(기본값 3600초)값 동안 Lambda호출없이 처리하게 됩니다.
(현재 Maximum TTL은 3600초이며, 그 이상 넘어갈수 없으며, 0초로 만들어서 캐쉬를 없앨수도 있습니다.)

### Create Custom Authorizer Lambda Function

새로 만드는 Lambda Function이 AWS의 다른 서비스를 호출한다면, 먼저 execution role 설정을 통해서 권한 부여가 필요합니다.


#### **API Gateway -> Lambda**

{% highlight json %}
{
    "type":"TOKEN",
    "authorizationToken":"<caller-supplied-token>",
    "methodArn":"arn:aws:execute-api:<regionId>:<accountId>:<apiId>/<stage>/<method>/<resourcePath>"
}
{% endhighlight %}

| Name | Description |
|:-----|:------------|
| authorizationToken | 클라이언트가 Api Gateway로 request의 header에 붙여서 보내는 Auth-Token의 개념 |
| type | payload type을 정의하며, 현재의 유일한 유효한 값은 "TOKEN" literal 하나입니다. |
| methodArn | API Gateway가 Lambda function에 값을 보내기전에 자동으로 넣어서 보냅니다. |

#### **Authorizer function in Lambda -> API Gateway**

Customer authorizer's Lambda function은 반드시 principal identifier 그리고 policy document를 포함한 response를 리턴시켜야 합니다.

{% highlight json %}

{
  "principalId": "xxxxxxxx",
  "policyDocument": {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": "execute-api:Invoke",
        "Effect": "Allow|Deny",
        "Resource": "arn:aws:execute-api:<regionId>:<accountId>:<appId>/<stage>/<httpVerb>/[<resource>/<httpVerb>/[...]]"
      }
    ]
  }
}

// Example
// GET Method를 Deny 시키는 예제 입니다.
{
  "principalId": "user",
  "policyDocument": {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": "execute-api:Invoke",
        "Effect": "Deny",
        "Resource": "arn:aws:execute-api:us-west-2:123456789012:ymy8tbxw7b/*/GET/"
      }
    ]
  }
}
{% endhighlight %}

| Name | Description |
| Effect | Allow, Deny로 해당 API Gateway의 Action을 실행시킬지 말지 결정합니다. |
| Action | Action은 Resource를 정의하는 API Gateway Execution Service입니다. |
| Resource | * (wild card)를 사용해서 Resource 를 정의할수 있습니다. |
| PrincipalId | $context.authorizer.principalId 변수를 사용해 mapping table 에 access할 수 있습니다. |

#### Create a Custom Authorization for API Methods

<img src="{{ page.static }}gateway_authorizer.png" class="img-responsive img-rounded">

<img src="{{ page.static }}custom-auth-set-authorizer-on-method.png" class="img-responsive img-rounded">


# Mapping Templates

API Gateway는 Request, Response 데이터를 Backend 그리고 Client에 맞게끔 변환시켜줄 수 있으며, 또한 Validation의 기능이 있습니다.

### Models

{% highlight json %}
{
  "department": "produce",
  "categories": [
    "fruit",
    "vegetables"
  ],
  "bins": [
    {
      "category": "fruit",
      "type": "apples",
      "price": 1.99,
      "unit": "pound",
      "quantity": 232
    },
    {
      "category": "fruit",
      "type": "bananas",
      "price": 0.19,
      "unit": "each",
      "quantity": 112
    },
    {
      "category": "vegetables",
      "type": "carrots",
      "price": 1.29,
      "unit": "bag",
      "quantity": 57
    }
  ]
}
{% endhighlight %}

위의 JSON Data는 다음과 같은 Model로 정의될수 있습니다. 



{% highlight json %}
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "title": "GroceryStoreInputModel",
  "type": "object",
  "properties": {
    "department": { "type": "string" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    },
    "bins": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "category": { "type": "string" },
          "type": { "type": "string" },
          "price": { "type": "number" },
          "unit": { "type": "string" },
          "quantity": { "type": "integer" }
        }
      }
    }
  }
}
{% endhighlight %}

| Name | Description |
| $schema | JSON Schema version 을 나타냅니다.  |
| title | 사람이 읽을수 있는 Identifier 입니다. |
| type | object, array, string, number, integer 등이 들어갈수 있습니다. |
| properties | type이 object이면 안에 들어가는 내용물들 | 

또한 추가적으로 minimum, maximum, string lengths, numeric values, array item lengths, regular expressions 등등을
더 추가해줄수 있습니다.

### Mapping Templates

Mapping Templates은 data 를 다른 형식으로 변환하는데 사용이 됩니다. 
Mapping Templates을 정의하기 위해서 API Gateway는 [Velocity Template Language][Velocity] 
또는 [JsonPath Expressions][JSON Path]을 사용합니다. 
Input mapping Templates 그리고 Output mapping templates을 각각 따로 만들어줘야 합니다.
아래의 예제는 JSON데이터를 받아서 JSON으로 만들어주는 ㅡㅡ;; 예제입니다.

{% highlight bash %}
#set($inputRoot = $input.path('$'))
{
  "department": "$inputRoot.department",
  "categories": [
#foreach($elem in $inputRoot.categories)
    "$elem"#if($foreach.hasNext),#end
        
#end
  ],
  "bins" : [
#foreach($elem in $inputRoot.bins)
    {
      "category" : "$elem.category",
      "type" : "$elem.type",
      "price" : $elem.price,
      "unit" : "$elem.unit",
      "quantity" : $elem.quantity
    }#if($foreach.hasNext),#end
        
#end
  ]
}
{% endhighlight %}


[Velocity]: http://velocity.apache.org/engine/devel/vtl-reference.html
[JSON Path]: http://goessner.net/articles/JsonPath/