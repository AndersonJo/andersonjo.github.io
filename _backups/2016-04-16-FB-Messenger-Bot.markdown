---
layout: post
title:  "Facebook Messenger Bot"
date:   2016-04-16 01:00:00
categories: "facebook"
static: /assets/posts/FB-Messenger-Bot/
tags: ['facebook', 'chatting', 'bot', 'hubot', 'node.js', 'express', 'ejs']
---

<header>
<img src="{{ page.static }}facebook-messenger-800.jpg" class="img-responsive img-rounded" style="width:100%">
</header>

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

### Configure Webhooks

[FB Messenger Console][FB Messenger Console]

현재 Facebook Messenger는 마케팅, 프로모션으로 사용을하면 안됩니다.<br>
따라서 FB에서는 이를 방지하기 위해서 IOS App처럼  Approved를 받아야 합니다.

Faceook Developer에서 Website App을 만든후 콘솔로 들어옵니다.<br>
Messenger -> Setup Webhooks 를 눌러서 설정을 해줍니다.

이때 Callback URL은 **HTTPS**만 사용가능합니다.

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

서버단에서는 다음과 같이 합니다.

{% highlight javascript %}
router.get('/webhook/', function(req, res){
  if (req.query['hub.verify_token'] == config.FB_VALICATION_TOKEN){
    return res.send(req.query['hub.challenge']);
  };
  return res.send('Error, wrong validation token');
});
{% endhighlight %}


정확하게 webhook을 걸고, Verification까지 완성되면 다음과 같이 됩니다.

<img src="{{ page.static }}webhook_complete.png" class="img-responsive img-rounded">

### Subscribe the app to the page

다음으로 페이스북 페이지를 만들어줍니다. (네.. 못생긴거 압니다.. 죄송합니다.) <br>
[FB Create Page][FB Create Page]

<img src="{{ page.static }}anderson_page.png" class="img-responsive img-rounded">

FB 콘솔안의 Token Generation에서 만들어준 Page를 설정해줍니다.

<img src="{{ page.static }}page_conf.png" class="img-responsive img-rounded">

curl을 통해서 해당 Page를 App이 subscribe하도록 설정해줍니다.

{% highlight bash %}
curl -ik -X POST "https://graph.facebook.com/v2.6/me/subscribed_apps?access_token=<token>"
{% endhighlight %}


### Receive Messages

{% highlight javascript %}
router.post('/webhook/', function (req, res) {
  messaging_events = req.body.entry[0].messaging;
  for (i = 0; i < messaging_events.length; i++) {
    event = req.body.entry[0].messaging[i];
    sender = event.sender.id;
    if (event.message && event.message.text) {
      text = event.message.text;
      // Handle a text message from this sender
    }
  }
  res.sendStatus(200);
});
{% endhighlight %}

페이스북 페이지에 들어가서 메세지를 열고 메세지를 보냅니다. 

<img src="{{ page.static }}send_msg.png" class="img-responsive img-rounded">

req.body는 다음과 같이 생겼습니다.

{% highlight javascript %}
{ object: 'page',
  entry: 
    [ { id: 1696327967283521,
        time: 1461911522431,
        messaging: [ 
          { sender: { id: 1118282011555900 },
            recipient: { id: 1696327967283521 },
            timestamp: 1461911522401,
            message:  { mid: 'mid.1461911522394:5daed2bac5cfd3cf54',
                        seq: 16,
                        text: '하이! 앤더슨 조!' } }] }] }
{% endhighlight %}


### Send Messages

{% highlight javascript %}
function sendTextMessage(sender, text) {

  var messageData = {
    text:text
  }

  request({
    url: 'https://graph.facebook.com/v2.6/me/messages',
    qs: {access_token:token},
    method: 'POST',
    json: {
      recipient: {id:sender},
      message: messageData,
    }
  }, function(error, response, body) {
    if (error) {
      console.log('Error sending message: ', error);
    } else if (response.body.error) {
      console.log('Error: ', response.body.error);
    }
  });
}

function sendGenericMessage(sender) {
  messageData = {
    "attachment": {
      "type": "template",
      "payload": {
        "template_type": "generic",
        "elements": [{
          "title": "First card",
          "subtitle": "Element #1 of an hscroll",
          "image_url": "http://messengerdemo.parseapp.com/img/rift.png",
          "buttons": [{
            "type": "web_url",
            "url": "https://www.messenger.com/",
            "title": "Web url"
          }, {
            "type": "postback",
            "title": "Postback",
            "payload": "Payload for first element in a generic bubble",
          }],
        },{
          "title": "Second card",
          "subtitle": "Element #2 of an hscroll",
          "image_url": "http://messengerdemo.parseapp.com/img/gearvr.png",
          "buttons": [{
            "type": "postback",
            "title": "Postback",
            "payload": "Payload for second element in a generic bubble",
          }],
        }]
      }
    }
  };
  request({
    url: 'https://graph.facebook.com/v2.6/me/messages',
    qs: {access_token:token},
    method: 'POST',
    json: {
      recipient: {id:sender},
      message: messageData,
    }
  }, function(error, response, body) {
    if (error) {
      console.log('Error sending message: ', error);
    } else if (response.body.error) {
      console.log('Error: ', response.body.error);
    }
  });
}
{% endhighlight %}


### Final Results

<img src="{{ page.static }}capture1.png" class="img-responsive img-rounded">

<img src="{{ page.static }}capture2.png" class="img-responsive img-rounded">

<img src="{{ page.static }}capture3.png" class="img-responsive img-rounded">

<img src="{{ page.static }}capture4.png" class="img-responsive img-rounded">



[FB Messenger Console]: https://developers.facebook.com/apps/879182215524653/messenger/
[FB Create Page]: https://www.facebook.com/pages/create/