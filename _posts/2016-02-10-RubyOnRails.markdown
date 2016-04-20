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


# Unicorn

### Gemfile

Gemfile 안에 다음과 같이 추가시킵니다.

{% highlight bash %}
gem 'unicorn'
gem "unicorn-rails"
{% endhighlight %}

### config/unicorn.rb

{% highlight ruby %}
# Set the current app's path for later reference. Rails.root isn't available at
# this point, so we have to point up a directory.
app_path = File.expand_path(File.dirname(__FILE__) + '/..')
working_directory app_path

# config/unicorn.rb
worker_processes 5
preload_app true
timeout 20

# Set up socket location
# listen "sockets/unicorn.sock", :backlog => 64
# listen app_path + '/tmp/unicorn.sock', backlog: 64

# For development, you may want to listen on port 3000 so that you can make sure
# your unicorn.rb file is soundly configured.
# listen(3000, backlog: 64) if ENV['RAILS_ENV'] == 'development'

# Logging
# stderr_path "./log/unicorn.stderr.log"
# stdout_path "./log/unicorn.stdout.log"

# Set master PID location
pid "./pids/unicorn.pid"


before_fork do |server, worker|
  Signal.trap 'TERM' do
    puts 'Unicorn master intercepting TERM and sending myself QUIT instead'
    Process.kill 'QUIT', Process.pid
  end

  defined?(ActiveRecord::Base) and
      ActiveRecord::Base.connection.disconnect!
end

after_fork do |server, worker|
  Signal.trap 'TERM' do
    puts 'Unicorn worker intercepting TERM and doing nothing. Wait for master to send QUIT'
  end

  defined?(ActiveRecord::Base) and
      ActiveRecord::Base.establish_connection
end
{% endhighlight %}

### Running Rails as Unicorn

{% highlight bash %}
rails server unicorn
{% endhighlight %}

위와 같이 하면 unicorn으로 실행이 되지만, Gemfile안에 unicorn-rails를 추가시키면 <br>
기본 서버가 unicorn으로 설정이 되서 rails s 만 해줘도 unicorn으로 실행됩니다.


# Logger

#### **STDOUT**

environment.rb 파일에 다음을 추가시키면 로그가 다 떨어져서 나온다.

{% highlight ruby %}
if Rails.env.development?
  Rails.logger = Logger.new(STDOUT)
  ActiveRecord::Base.logger = nil # Turn off SQL Query Logging
end
{% endhighlight %}


