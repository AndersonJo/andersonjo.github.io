---
layout: post
title:  "Facebook Messenger Bot"
date:   2016-04-16 01:00:00
categories: "facebook"
static: /assets/posts/FB-Messenger-Bot/
tags: ['facebook', 'chatting', 'bot', 'hubot', 'node.js', 'express', 'ejs']
---

<img src="{{ page.static }}facebook-messenger-800.jpg" class="img-responsive img-rounded">


# Setting Node.js & Express

### Using HTML instead of Jade

EJS를 사용하며 app.js 에 다음과 같이 설정을 합니다.

{% highlight javascript %}
// view engine setup
app.engine('html', require('ejs').renderFile);
app.set('view engine', 'html');
app.set('views', path.join(__dirname, 'views'));
{% endhighlight %}

packages.js 의 dependencies에는 ejs를 추가 시킵니다.<br>

{% highlight javascript %}
{
    "scripts": {
        "start": "NODE_PATH=./ && supervisor --watch . ./bin/www"
      },
    "dependencies": {
        ...
        "ejs": "*"
      }
}
{% endhighlight %}  

router에서는 res.render를 통해서 views/index.html 에 있는 html rendering할 수 있습니다. 

{% highlight javascript %}
router.get('/', function (req, res, next) {
    res.render('index', {title: 'Facebook Messenger Bot Tutorial'});
});
{% endhighlight %}  

npm start를 실행하면, packages.js의 scripts에 있는 start가 실행이 됩니다.

### SSL & Nginx

{% highlight bash %}
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/nginx/ssl/nginx.key -out /etc/nginx/ssl/nginx.crt
{% endhighlight %}





# Facebook Messenger API

### Introduction

[Messenger Introduction][Messenger Introduction]

현재 Facebook Messenger는 마케팅, 프로모션으로 사용을하면 안됩니다.<br>
따라서 FB에서는 이를 방지하기 위해서 IOS App처럼  Approved를 받아야 합니다.

Faceook Developer에서 Website App을 만든후 콘솔로 들어옵니다.<br>
Messenger -> Setup Webhooks 를 눌러서 설정을 해줍니다.

이때 Callback URL은 HTTPS만 사용가능합니다.

<img src="{{ page.static }}messenger-webhook.png" class="img-responsive img-rounded">

https://domain.co.kr/webhook/ 같은 주소를 사용하고<br>
Token은 "anderson_jo_validation_token" 처럼 맘대로 넣으면 됩니다.<br>
Nginx 설정은 다음과 같이 했습니다.

{% highlight nginx %}
upstream fb-msg{
    server localhost:3000;
}

server{
    listen 443;
    server_name domain.co.kr;
    charset utf-8;
    client_max_body_size 25M;

    ssl on;
    ssl_certificate     /etc/nginx/ssl/ssl.pem;
    ssl_certificate_key /etc/nginx/ssl/star_ssl.key;

    location / {
        proxy_redirect off;
        proxy_set_header   X-Real-IP            $remote_addr;
        proxy_set_header   X-Forwarded-For  $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_set_header   Host                   $http_host;
        proxy_set_header   X-NginX-Proxy    true;
        proxy_set_header   Connection "";
        proxy_http_version 1.1;
        proxy_pass         http://fb-msg;
    }
}
{% endhighlight %}



[Messenger Introduction]: https://developers.facebook.com/apps/879182215524653/messenger/