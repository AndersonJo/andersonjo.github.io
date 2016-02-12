---
layout: post
title:  "Ruby On Rails 101"
date:   2016-02-10 01:00:00
categories: "ruby"
static: /assets/posts/RubyOnRails101/
tags: ['/etc/init.d', 'ubuntu', 'service']
---

<img src="{{ page.static }}rubyrails.png" class="img-responsive img-rounded">

# Logger

#### **STDOUT**

environment.rb 파일에 다음을 추가시키면 로그가 다 떨어져서 나온다.

{% highlight ruby %}
if Rails.env.development?
  Rails.logger = Logger.new(STDOUT)
end
{% endhighlight %}