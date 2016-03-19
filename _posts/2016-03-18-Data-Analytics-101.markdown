---
layout: post
title:  "Data Analytics 101"
date:   2016-03-18 01:00:00
categories: "analytics"
static: /assets/posts/DataAnalytics101/
tags: ['python', 'data analytics', 'r']
---


<img src="{{ page.static }}analytics.jpg" class="img-responsive img-rounded">

# Installation

### Example

예제는 Python Matplotlib의 plot을 보는 예제

<img src="{{ page.static }}pylab.png" class="img-responsive img-rounded">

### Installing Jupyter

쥬피터는 Ipython Notebook에서 더 발전된 버젼으로 Python, R, Scala등의 데이터 분석에 쓰이는 언어들을 선택해서 웹애플리케이션으로
사용이 가능하게 해줍니다.

{% highlight bash %}
sudo pip install jupyter
jupyter notebook
{% endhighlight %}

### 한글 설정

문서의 가장 윗쪽에 다음과 같이 설정합니다.

{% highlight python %}
#-*- coding:utf-8 -*-
%pylab inline
matplotlib.rc('font', family='NanumGothic')
{% endhighlight %}

만약 내가 갖고 있는 모든 폰트들을 열고 싶다면..

{% highlight python %}
import matplotlib.font_manager
print [f.name for f in matplotlib.font_manager.fontManager.ttflist]
{% endhighlight %}



# Chart Tutorial

### Bar Charts

{% highlight python %}
#-*- coding:utf-8 -*-
%pylab inline
matplotlib.rc('font', family='NanumGothic')

names = [u'창민', u'정아', u'윤서', u'미정', u'세준']
cookies = [5, 11, 7, 1, 13]
xs = range(1, len(cookies)+1)

plt.bar(xs, cookies, color='#ff3366')
plt.ylabel('Number of Cookies')
plt.title("Cookie?")
plt.xticks(xs, names)
{% endhighlight %}

<img src="{{ page.static }}barchart.png" class="img-responsive img-rounded">

### Histogram

{% highlight python %}
grades = np.random.standard_gamma(100, size=1000)
plt.hist(grades, bins=10, color='red')
{% endhighlight %}

<img src="{{ page.static }}histogram.png" class="img-responsive img-rounded">