---
layout: post
title:  "AWS IOT Tutorial"
date:   2016-09-22 01:00:00
categories: "aws"
asset_path: /assets/posts2/AWS/
tags: ['MQTT', 'Socket']

---

<header>
    <img src="{{ page.asset_path }}awsiot-how-it-works_HowITWorks_1-26.png" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small>빅데이터 + IOT + 인공지능 = AWESOME        
    </small>
    </div>
</header>

# Node.js <-(MQTT)-> AWS IOT

### Create Certificate

먼저 **Create a certificate** 버튼을 눌러서 certificate을 만듭니다. AWS IOT에서는 반드시(must) secure network만으로 통신을 해야 합니다.<br>
아래의 public key, private key, certificate, 그리고 root CA를 다운로드 받습니다. 

<img src="{{ page.asset_path }}iot-create-certificate.png" class="img-responsive img-rounded">

### Create a rule

topic_2를 watch합니다. 이때 JSON의 데이터중 temperature 가 50 이상이면 Action을 취하게 됩니다.<br>
아래의 예제에서는 topic_1으로 데이터를 republish 합니다. <br>
clientId부분에서 AWS Account (현재 내가 쓰는 계정안에서) unique해야 하는데.. 만약 동일한 다른 thing이 이미 접속해있고.. 
이상태에서 다른 동일한 clientId를 갖은 thing 이 접속하려하면, 기존에 접속해 있던  thing 은 terminate되게 됩니다.
(서로 접속하고 떨어지고 접속하고 떨어지고 반복하게 됨)

<img src="{{ page.asset_path }}iot-create-rule.png" class="img-responsive img-rounded">

### Device Example in Node.js

AWS IOT Device SDK for Node.js 는 다음과 같이 설치할수 있습니다. (현재 C, Node.js SDK 제공)

{% highlight bash %}
npm install aws-iot-device-sdk
{% endhighlight %}


certificate, private key, [root-CA.crt][Root CA]를 certs 디렉토리 안으로 복사해줍니다.
그리고 awsIot.device의 constructor부분을 변경해줍니다.

{% highlight javascript %}
var awsIot = require('aws-iot-device-sdk');

var device = awsIot.device({
    "host": "aliokj1e7vx6m.iot.ap-northeast-2.amazonaws.com",
    "port": 8883,
    "clientId": "anderson",
    "thingName": "anderson",
    "caCert": "certs/root-CA.crt",
    "clientCert": "certs/90212d7bd4-certificate.pem.crt",
    "privateKey": "certs/90212d7bd4-private.pem.key"
});

// Device is an instance returned by mqtt.Client(), see mqtt.js for full documentation
device.on('connect', function () {
    device.subscribe('topic_1');
    device.publish('topic_2', JSON.stringify({message: 'Hello Anderson', temperature: 60}));
});

device.on('message', function (topic, payload) {
    console.log('message received:', topic, payload.toString());
});
{% endhighlight %}

위의 코드를 실행시키면 다음과 같이 결과가 나옵니다.

{% highlight bash %}
$ node device.js
message received: topic_1 {"message":"Hello Anderson","temperature":60}
{% endhighlight %}

### Shadow Example in Node.js 

{% highlight javascript %}
var awsIot = require('aws-iot-device-sdk');

var thingShadows = awsIot.thingShadow({
    "host": "aliokj1e7vx6m.iot.ap-northeast-2.amazonaws.com",
    "port": 8883,
    "clientId": "anderson",
    "thingName": "anderson",
    "keyPath": "certs/90212d7bd4-private.pem.key",
    "certPath": "certs/90212d7bd4-certificate.pem.crt",
    "caPath": "certs/root-CA.crt",
    "region": "ap-northeast-2"
});

// Client token value returned from thingShadows.update() operation
var clientTokenUpdate;

// Simulated device values
var rval = 187;
var gval = 114;
var bval = 222;

thingShadows.on('connect', function () {
    thingShadows.register('RGBLedLamp');
    
    setTimeout(function () {
        var rgbLedLampState = {"state": {"desired": {"red": rval, "green": gval, "blue": bval}}};
        clientTokenUpdate = thingShadows.update('RGBLedLamp', rgbLedLampState);

        if (clientTokenUpdate === null) {
            console.log('update shadow failed, operation still in progress');
        }
    }, 5000);
});
{% endhighlight %}


**thingShadows.register("RGBLedLamp")** 함수를 통해서 RGBLedLamp라는 Shadow 정보를 watch하겠다는 뜻입니다.<br>
이를 통해서 get, update, delete시에 client에서 event를 받을 수 있습니다. 

**setTimeout** 함수를 통해 5초뒤에 가장 최신의 상태로 RGBLedLamp shadow를 업데이트 시킵니다.<br>
처음 update시에 5초동안 기다리는 것은 필수적입니다. Thing Shadow registration은 delay를 요하기 때문입니다.


{% highlight javascript %}
thingShadows.on('status',
    function (thingName, stat, clientToken, stateObject) {
        console.log('received: ' + stat + ' on ' + thingName + ': ' + JSON.stringify(stateObject));
    });
{% endhighlight %}

update(), get, delete() 시에 report값이 옵니다. 아래는 stateObject 의 값입니다.


{% highlight json %}
{
  "timestamp": 1474530541,
  "state": {
    "red": 187,
    "green": 114,
    "blue": 222
  },
  "metadata": {
    "red": {
      "timestamp": 1474530541
    },
    "green": {
      "timestamp": 1474530541
    },
    "blue": {
      "timestamp": 1474530541
    }
  }
}
{% endhighlight %}





[Root CA]: https://www.symantec.com/content/en/us/enterprise/verisign/roots/VeriSign-Class%203-Public-Primary-Certification-Authority-G5.pem