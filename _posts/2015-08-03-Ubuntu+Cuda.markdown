---
layout: post
title:  "Ubuntu 15.04 + GTX960"
date:   2015-08-03 02:00:00
categories: "ubuntu"
asset_path: /assets/posts/GTX960+Ubuntu15.04+Hello-World/
---

이번에 GTX960 그래픽카드를 질렀습니다.<br> 
설치 환경은 Ubuntu 15.04 + GTX960 인데, 혹시 나중에 다시 보게 될까봐 여기에다가 적습니다.

<img src="{{page.asset_path}}gtx960.jpg" class="img-responsive img-rounded">

### ACPI PPC Probe failed

GTX960 디바이스를 읽지 못해서 생기는 에러입니다. <br>
그냥 warning 정도의 에러인데.. 이것과 상관없이.. 화면이 보이지 않는다면.. <br>
**nomodeset** 옵션을 주고 우분투를 설치또는 로그인하면 됩니다.
 
> 우분투 설치시에는 F6 (other options)를 눌러서 옵션을 지정할수 있습니다.

### 32bit Libraries (Optional)
GTX960 Driver를 설치하기전, 32bit 라이브러리를 설치해줍니다. <br>
해당 라이브러리는 또한 Android Studio사용시 설치 가능하게 해줍니다. 

{% highlight bash%}
sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32z1 lib32z1-dev
{% endhighlight %}


### Oracle Java 

{% highlight bash%}
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java7-installer
{% endhighlight %}


### 벼루 (Optional)

{% highlight bash%}
sudo apt-get install uim uim-byeoru
uim-pref-gtk
{% endhighlight %}

### XClip (Optional)

{% highlight bash%}
sudo apt-get install xclip
{% endhighlight %}

### Open Nautilus (Ubuntu 14.04)

우분투 14.x 버젼에서만...

{% highlight bash%}
sudo apt-get install nautilus-open-terminal
{% endhighlight %}

### Command Prompt 설정 (Optional)

.bashrc 파일에 추가
*하둡, CUDA, Android, Java, Spark설정 모두 들어가있기 때문에 따로 설정 필요합니다.*

{% highlight bash%}
parse_git_branch() {

    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'

}
export PS1='\[\033[00;36m\]\u:\[\033[0;33m\]\W$(parse_git_branch)>\[\033[00m\]'

export ANDROID_HOME=/home/anderson/apps/android-sdk
export JAVA_HOME=/usr/lib/jvm/java-7-oracle
export HADOOP_HOME=/usr/local/hadoop-2.7.2
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_CONF_DIR=$HADOOP_HOME/conf
export HADOOP_CLASSPATH=$HADOOP_HOME/conf
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib/native"
export HIVE_HOME="/usr/local/hive"
export DERBY_HOME=/usr/local/derby
export CUDAHOME=/usr/local/cuda-7.5

export CLASSPATH=$CLASSPATH:$HIVE_HOME/lib/*:.
export CLASSPATH=$CLASSPATH:$HADOOP_HOME/lib/*:.
export CLASSPATH=$CLASSPATH:$DERBY_HOME/lib/derby.jar
export CLASSPATH=$CLASSPATH:$DERBY_HOME/lib/derbytools.jar

export PATH=$PATH:$HADOOP_HOME/bin
export PATH=$PATH:$HADOOP_HOME/sbin
export PATH=$PATH:$CUDAHOME/bin
export PATH=$PATH:$ANDROID_HOME:$ANDROID_HOME/platform-tools
export PATH=$PATH:$HIVE_HOME/bin
export PATH=$PATH:$DERBY_HOME/bin
unset JAVA_TOOL_OPTIONS

{% endhighlight %}


### Sublime (Optional)

**GCC Configuration**

~/.config/sublime-text-3/Packages/User 에다가 넣으면 됩니다.

[c.sublime-build][c.sublime]

{% highlight json %}
{
	"cmd" : ["gcc ${file_name} -o ${file_base_name} && ${file_path}/${file_base_name}"],
	"shell" : true,
	"selector" : "source.c",
	"working_dir" : "$file_path",
}
{% endhighlight %}

**Package Controller**

Ctrl + ` 후 다음을 붙여넣기 합니다.

{% highlight python %}
import urllib.request,os,hashlib; h = '2915d1851351e5ee549c20394736b442' + '8bc59f460fa1548d1514676163dafc88'; pf = 'Package Control.sublime-package'; ipp = sublime.installed_packages_path(); urllib.request.install_opener( urllib.request.build_opener( urllib.request.ProxyHandler()) ); by = urllib.request.urlopen( 'http://packagecontrol.io/' + pf.replace(' ', '%20')).read(); dh = hashlib.sha256(by).hexdigest(); print('Error validating download (got %s instead of %s), please try manual install' % (dh, h)) if dh != h else open(os.path.join( ipp, pf), 'wb' ).write(by)
{% endhighlight %}

Ctrl + Shift + p 누른 후 Package control: Install Package 명령어로 Package 설치


### Ruby IRC (Optional)

~ 위치에 .irbrc 파일에 다음을 저장합니다.

{% highlight ruby %}
require 'irb/completion'
require 'map_by_method'
require 'what_methods'
require 'pp'
IRB.conf[:AUTO_INDENT]=true
{% endhighlight %}

### RVM (Optional)

특정 버젼을 자동 실행합니다.
{% highlight bash %}
bash -l -c 'rvm use ruby-dev-2.1.0'
{% endhighlight %}



### Postgres

설치..
{% highlight bash %}
sudo apt-get install postgresql-client
sudo apt-get install postgresql postgresql-contrib
apt-cache search postgres
{% endhighlight %}

postgres 패스워드 변경
{% highlight text %}
sudo -u postgres psql postgres
\password postgres
{% endhighlight %}

peer -> md5 로 고친다.

{% highlight bash %}
sudo vi /etc/postgresql/9.4/main/pg_hba.conf
local   all             postgres                                md5
local   all             all                                     md5
sudo service postgresql restart
{% endhighlight %}

### MySQL

{% highlight text %}
sudo vi /etc/mysql/mysql.conf.d/mysqld.cnf

[mysqld_safe]
default-character-set=utf8

[mysqld]
collation-server = utf8_unicode_ci
init-connect='SET NAMES utf8'
character-set-server = utf8
{% endhighlight %}



### Current Nvidia Card

현재 그래픽 카드 모델을 알고 싶을때는...
{% highlight bash%}
lspci -vnn | grep -i VGA -A 12
{% endhighlight %}



### CUDA Toolkit

CUDA Toolkit설치시 GPU Drive, CUDA, Nsight 등이 전부다 깔림니다.<br>
아래의 주소에서 RUN파일을 다운로드 받습니다.<br>
[https://developer.nvidia.com/cuda-downloads][cuda-toolkit]

1. 다운받은 폴더로 들어갑니다.
2. chmod로 실행파일로 바꿔줍니다.
3. CTRL + ALT + F3 
4. 로그인
5. init 3
6. sudo service lightdm stop
7. sudo su
8. ./NVIDIA*.run 파일 실행
9. reboot


### CUDA Testing

Cuda샘플이 설치된 환경으로 이동한다면...

{% highlight bash%}
cd ./1_Utilities/deviceQuery
make
./deviceQuery
{% endhighlight %}


파일이 잘 실행이 되는지 확인을 합니다.


### Saving a new X Configuration

Nvidia 드라이버 설치시 자동으로 해주긴 하지만.. 혹시 새롭게 다시 재정의 필요시 다음의 명령어를 실행시켜줍니다.

{% highlight bash%}
sudo nvidia-xconfig
{% endhighlight %}


### 검은화면, Low 그래픽 화면.. 에러

에러가 일어났을 경우에만.. 다음의 라이브러리들을 설치합니다.

{% highlight bash%}
sudo apt-get install dkms fakeroot build-essential linux-headers-generic
{% endhighlight %}

[c.sublime]: {{ page.asset_path }}c.sublime-build
[gtx-driver]: http://www.geforce.com/drivers
[cuda-toolkit]: https://developer.nvidia.com/cuda-downloads
[nvidia-download]: http://www.nvidia.com/Download/index.aspx