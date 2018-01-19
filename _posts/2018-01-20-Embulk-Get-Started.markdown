---
layout: post
title:  "Embulk + Google Cloud"
date:   2017-11-25 01:00:00
categories: "etl"
asset_path: /assets/images/
tags: ['google', 'gcm', 'bigquery']
---

# Embulk Installation

Embulk는 [git repository](https://github.com/embulk/embulk)에서 코드를 받을 수 있습니다.<br>
설치는 다음과 같이 합니다.

{% highlight bash %}
curl --create-dirs -o ~/.embulk/bin/embulk -L "https://dl.embulk.org/embulk-latest.jar"
chmod +x ~/.embulk/bin/embulk
echo 'export PATH="$HOME/.embulk/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
{% endhighlight %}

# Embulk Tutorial

`embulk example`을 이용하면 샘플 CSV가 생성이 되고, 이것을 갖고서 tutorial을 진행할 수 있습니다.<br>
`embulk guess`를 실행하면, embulk는 데이터에 대해서 간단하게 훓어봅니다. 이후 configuration에서 빠진 부분들을 `추측`해서 채워넣게 됩니다.
따라서 사용자는.. 일단 대충 중요한 부분만 적어놓고, 나머지 설정에 들어가야 하는 내용들은 guess가 자동으로 설정해줄수 있습니다.<br>
`embulk preview`를 하게되면 데이터에 대해서 간단하게 확인을 해볼수 있습니다. 또한 설정한 것이 잘 돌아가는지도 확인도 해볼 수 있습니다.<br>
`embulk run config.yml`를 실행하게 되면 설정된 명령되로 embulk는 실제 일을 수행하게 됩니다.
여기서 데이터는 RDBMS에서 가져와서 Google Cloud에 들어갈 수도 있고, 다른 RDBMS또는 S3등.. 다양한 방법으로 데이터를 수정/배포 할 수 있습니다.



{% highlight bash %}
embulk example ./try1
embulk guess ./try1/seed.yml -o config.yml
embulk preview config.yml
embulk run config.yml
{% endhighlight %}