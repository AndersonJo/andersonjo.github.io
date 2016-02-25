---
layout: post
title:  "Ruby On Rails 101"
date:   2016-02-10 01:00:00
categories: "ruby"
static: /assets/posts/RubyOnRails101/
tags: ['/etc/init.d', 'ubuntu', 'service']
---

<img src="{{ page.static }}rubyrails.png" class="img-responsive img-rounded">

# Installation

### Install RVM

먼저 RVM을 설치합니다. (https://rvm.io/)<br>
먼저 콘솔을 열고 Edit -> Profile Preferences 누른후 2번째 탭에서 Run command as a login shell을 선택해줍니다.

{% highlight bash %}
gpg --keyserver hkp://keys.gnupg.net --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3
\curl -sSL https://get.rvm.io | bash -s stable
rvm install ruby-2.1.0
rvm list
rvm use ruby-2.1.0
{% endhighlight %}


### Install Rails

{% highlight bash %}
gem install rails -v 4.1.13
{% endhighlight %}

# Logger

#### **STDOUT**

environment.rb 파일에 다음을 추가시키면 로그가 다 떨어져서 나온다.

{% highlight ruby %}
if Rails.env.development?
  Rails.logger = Logger.new(STDOUT)
  ActiveRecord::Base.logger = nil # Turn off SQL Query Logging
end
{% endhighlight %}