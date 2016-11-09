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

### Installing LIRE Plugin

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

설치된 Plugins들을 확인합니다.

[http://localhost:9200/_nodes?plugin=true](http://localhost:9200/_nodes?plugin=true)

{% highlight json %}
"plugins": [
    {
        "name": "image",
        "version": "1.2.0",
        "description": "Elasticsearch Image Plugin",
        "jvm": true,
        "site": false
    }
]
{% endhighlight %}

### Python Client

Python Client를 설치합니다.

{% highlight bash %}
sudo pip install 'elasticsearch>=1.0.0,<2.0.0'
{% endhighlight %}

# Image Searching

### Mappings

먼저 시작에 앞서 아래의 코드와 같은 Mapping을 갖은 Index를 만들어야 합니다.

{% highlight python %}
mapping = {
    'mappings': {
        'test': {
            "properties": {
                "name": {
                    "type": "string"
                },
                "image": {
                    "type": "image",
                    "feature": {
                        "CEDD": {
                            "hash": [
                                "BIT_SAMPLING",
                                "LSH"
                            ]
                        },
                        "JCD": {
                            "hash": [
                                "BIT_SAMPLING",
                                "LSH"
                            ]
                        },
                        "FCTH": {}
                    },
                    "metadata": {
                        "jpeg.image_width": {
                            "type": "string",
                            "store": "yes"
                        },
                        "jpeg.image_height": {
                            "type": "string",
                            "store": "yes"
                        }
                    }
                }
            }
        }
    }
}

es.indices.create(index='images', ignore=400, body=mapping)
{% endhighlight %}

### Indexing

{% highlight python %}
img = open('../data/images/rainbowsix.jpg', 'rb').read()

body = {
    'name': 'rainboxsix.jpg',
    'image': base64.b64encode(img)
}
# es.create(index='images', doc_type='test', body=body)
es.index(index='images', doc_type='test', id=1, body=body)
{% endhighlight %}


### Searching

{% highlight python %}
img2 = open('../data/images/rainbosix_cropped.jpg', 'rb').read()

es.search(index='images', body ={
        'fields': ['name'],
        'query': {
            'image': {
                'image': {
                    'feature': 'JCD',
                    'image': base64.b64encode(img2),
                    'hash': 'LSH',
                    'limit': 10
                }
            }
        }
    })
{% endhighlight %}