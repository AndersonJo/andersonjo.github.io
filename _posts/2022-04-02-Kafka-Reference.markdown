---
layout: post 
title:  "ElasticSearch Reference"
date:   2022-04-02 01:00:00 
categories: "elasticsearch"
asset_path: /assets/images/ 
tags: ['kafka', 'elasticsearch', 'connector']
---

# ElasticSearch Reference

## Authentication

{% highlight bash %} {% raw %} $ PASSWORD=$(kubectl get secret elk-es-elastic-user -n elasticsearch -o
go-template='{{.data.elastic | base64decode}}')
{% endraw %} {% endhighlight %}

## Index

{% highlight bash %}

# List Index

$ curl -XGET -u "elastic:${PASSWORD}" -k "https://localhost:9200/_cat/indices"
{% endhighlight %}

## CRUD

**Create Data**

curl -XPUT "https://es.inthewear.com:9200/test-index/_doc/1" \
    -u "elastic:${PASSWORD}" \
    -H 'Content-Type: application/json' \
    -d '
{"name": "Anderson",
"data": {"name": "Matrix", "price": 2500.5},
"/api/path": {"name": "Twitter", "value": 23.5},
"created_at" : "2022-04-04T11:39:47.325156",
"created_timezone_at": "2022-04-04T11:39:06.140117+09:00"
}'
