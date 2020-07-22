---
layout: post
title:  "별것 아닌 문제에 대한 소소한 꿀팁"
date:   2020-04-25 01:00:00
categories: "etc"
asset_path: /assets/images/
tags: ['encoding', 'utf-8', 'unicode', 'nfd', 'nfc']
---

<header>
    <img src="{{ page.asset_path }}problem-bg.jpg" class="img-responsive img-rounded img-fluid center">
    <div style="text-align:right">
    <a style="background-color:black;color:white;text-decoration:none;padding:4px 6px;font-family:-apple-system, BlinkMacSystemFont, &quot;San Francisco&quot;, &quot;Helvetica Neue&quot;, Helvetica, Ubuntu, Roboto, Noto, &quot;Segoe UI&quot;, Arial, sans-serif;font-size:12px;font-weight:bold;line-height:1.2;display:inline-block;border-radius:3px" href="https://unsplash.com/photos/-2vD8lIhdnw" target="_blank" rel="noopener noreferrer" title="Download free do whatever you want high-resolution photos from Nathan Dumlao"><span style="display:inline-block;padding:2px 3px"><svg xmlns="http://www.w3.org/2000/svg" style="height:12px;width:auto;position:relative;vertical-align:middle;top:-2px;fill:white" viewBox="0 0 32 32"><title>unsplash-logo</title><path d="M10 9V0h12v9H10zm12 5h10v18H0V14h10v9h12v-9z"></path></svg></span><span style="display:inline-block;padding:2px 3px">Photo by JESHOOTS.COM on Unsplash</span></a>
    </div>
    
    
</header>

# 1. 별것 아닌 문제에 대한 소소한 꿀팁

업무를 하다보면.. 별것도 아닌데.. 시간을 지체하게 만드는 일들이 있다. <br>
알고 있으면 빠르게 해결되는데.. 검색하기도 참 난감한 문제들이 있다. <br>
그런 사소한 것들을 해당 페이지에 정리해보고자 한다 :)

# 2. 파이썬 소소한 팁 

## 2.1 한글 자음 모음 깨짐 현상 

맥, 우분투 그리고 윈도우 는 서로 한글을 사용할때의 인코딩이 틀려서 글자가 깨져서 읽혀지는 경우가 있습니다. <br>
utf-8 문제인가 해서 파일 열을때 encoding='utf-8' 같은 옵션을 주어도 해결이 안되는데.. 이건 다음과 같이 합니다. 

 - 윈도우, 우분투: NFC (Normal Form Composed) 
 - 맥: NFD (Normal Form Decomposed) <- 이녀석이 문제
 
### 2.1.1 윈도우,우분투 -> 맥

현재 실행 환경우 우분투인데..
맥에서 사용하는 파일 글자 형태로 바꾸면.. 자음 모음이 쪼개진다  
 
{% highlight python %}
from unicodedata import normalize
 
hello = '안녕하세요'
mac_hello = normalize('NFD', hello)

print([c for c in mac_hello])                                                                                                             
['ᄋ', 'ᅡ', 'ᆫ', 'ᄂ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄉ', 'ᅦ', 'ᄋ', 'ᅭ']
{% endhighlight %} 


### 2.1.2 맥 -> 윈도우,우분투

자음, 모음이 분리되서 읽히는 것이 정상적으로 한글자씩 읽혀진다 

{% highlight python %}
window_hello = normalize('NFC', mac_hello)              

print([c for c in window_hello])                                                                                                          
['안', '녕', '하', '세', '요']
{% endhighlight %} 


## 2.2 PIP mirror 를 카카오 저장소로 변경하기

pytorch 다운로드하는데 4시간 뜨길래.. 바꿈. <br>
카카오로 변경하면 pytorch 다운로드가 1분이면 끝남

{% highlight bash %}
mkdir ~/.pip
vi ~/.pip/pip.conf
{% endhighlight %}

pip.conf 에는 아래의 내용을 넣는다

{% highlight bash %}
[global]
index-url=http://ftp.daumkakao.com/pypi/simple
trusted-host=ftp.daumkakao.com
{% endhighlight %}


## 2.3 Numpy Precision 시각화

Precision의 시각화를 변경시킵니다. 

{% highlight bash %}
data = (np.random.rand(4, 4) * 0.0001).astype(np.float32)                                                                                  
array([[1.70037983e-05, 1.24925400e-05, 1.05073095e-05, 8.56156257e-05],
       [8.56784827e-05, 6.01017491e-05, 1.74893994e-05, 7.26617436e-05],
       [6.08324372e-05, 7.23426347e-05, 3.67687244e-05, 4.75522647e-05],
       [2.31287959e-05, 7.36061265e-05, 8.91206510e-05, 2.37407185e-05]],
      dtype=float32)

np.set_printoptions(formatter={'float_kind':'{:f}'.format}

array([[0.000182, 0.000602, 0.000214, 0.000582],
       [0.000020, 0.000617, 0.000936, 0.000013],
       [0.000513, 0.000052, 0.000895, 0.000348],
       [0.000484, 0.000960, 0.000140, 0.000403]], dtype=float32)
{% endhighlight %}