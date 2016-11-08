---
layout: post
title:  "Elasticsearch - Lucene Image REtrieval"
date:   2015-11-04 01:00:00
categories: "elastic"
asset_path: /assets/posts2/Elasticsearch/
tags: ['LIRE']
---
<header>
    <img src="{{ page.asset_path }}tokyo-street-after-work-wallpaper.jpg" class="img-responsive img-rounded">
</header>

LIRE (Lucene Image REtrieval)은 content based retrieval로서 index에서 유사한 이미지를 가져올수 있도록 도와주는 Elasticsearch의 플러그인입니다.

# Installation

유사 이미지 검색을 하기 위해서는 LIRE를 설치해야 합니다. 문제는 LIRE가 지원하는 Elasticsearch의 버젼과 차이가 있습니다.
현재 Plugin의 버젼은 1.2.0이 나왔는데, 해당 버젼이 지원하는 Elasticsearch의 버젼은 1.0.1밖에 되지 않습니다. 
(현재 Elasticsearch는 5.0.0까지 나온 상태입니다.)

| Image Plugin | elasticsearch | Release date |
|:-------------|:--------------|:-------------|
| 1.3.0-SNAPSHOT (master) | 1.1.0 |	
| 1.2.0 | [1.0.1](https://www.elastic.co/downloads/past-releases/elasticsearch-1-0-1) | 2014-03-20 |
| 1.1.0 | 1.0.1 | 2014-03-13 |
| 1.0.0 | 1.0.1	| 2014-03-05 |

Elasticsearch 1.0.1을 설치한이후, LIRE 플러그인을 설치합니다.

{% highlight bash %}
sudo /usr/share/elasticsearch/bin/plugin -install com.github.kzwang/elasticsearch-image/1.2.0
{% endhighlight %}