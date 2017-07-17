---
layout: post
title:  "Installing R on Ubuntu"
date:   2015-07-25 15:00:00
categories: "statistics"
tags: ['ubuntu', 'jupyter']
asset_path: /assets/images/
---

# Installing R

먼저  R을 설치합니다.

{% highlight bash %}
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'
sudo apt-get update
sudo apt-get install r-base
{% endhighlight %}

필수적인 libraries들을 설치합니다.

{% highlight bash %}
sudo -i R
{% endhighlight %}

### Installing Native R Kernel for Jupyter

{% highlight r %}
install.packages('devtools')
install.packages(c('repr', 'IRdisplay', 'evaluate', 'crayon', 'pbdZMQ', 'devtools', 'uuid', 'digest'))
devtools::install_github('IRkernel/IRkernel')
IRkernel::installspec()


{% endhighlight %}