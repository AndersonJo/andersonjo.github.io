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

{% highlight bash %}
{% raw %}
$ PASSWORD=$(kubectl get secret elk-es-elastic-user -n elasticsearch -o go-template='{{.data.elastic | base64decode}}')
{% endraw %}
{% endhighlight %}

## Index

{% highlight bash %}
# List Index
$ curl -XGET -u "elastic:${PASSWORD}" -k "https://localhost:9200/_cat/indices"
{% endhighlight %}
