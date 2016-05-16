---
layout: post
title:  "ROS Tutorial [Jade]"
date:   2015-09-21 01:00:00
categories: "robot"
asset_path: /assets/posts/ROS-Tutorial/
---
<div>
    <img src="{{ page.asset_path }}cturtle_ros.jpg" class="img-responsive img-rounded">
</div>

후엉~ 내가 이런것도 해볼줄이야 :) <br>
ROS는 Django가 웹개발을 빠르게 해주는 것 처럼, 로봇 프로그래밍을 빠르게 해줍니다.<br>
최근 꽤나 화제가 되었었던 [Lily][ref-lily]도 ROS로 만들어졌다고 합니다.




## <span class="glyphicon glyphicon-ok-sign" aria-hidden="true"></span> Workspace

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> Creating Workspace 

* [catin workspace][ref-catin]

먼저 ROS 환경설정을 잡아줘야 합니다. 

{% highlight bash %}
source /opt/ros/jade/setup.bash
{% endhighlight %}

워크스페이스를 만듭니다.

{% highlight bash %}
mkdir -p project_name/src
cd project_name/src
catkin_init_workspace

cd ..
catkin_make
source devel/setup.bash
printenv | grep ROS
{% endhighlight %}

이렇게 하면 build, devel, src 디렉토리가 만들어집니다.

## <span class="glyphicon glyphicon-ok-sign" aria-hidden="true"></span> Package

* [Package][ref-package]

package는 library같은 것이고, package.xml이 dependencies, version, maintainer등등의 메타 데이터를 저장합니다.<br>
package는 또한 OS의 여러군대에 퍼져있기 때문에, 각각의 패키지를 찾는데 ls, cd등은 불편할수 있기 때문에 Package Tools을 제공합니다.

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> find

{% highlight bash %}
rospack find package_name
# rospack find roscpp
{% endhighlight %}

find는 ROS_PACKAGE_PATH 환경 변수안에서만 찾습니다.

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> roscd

{% highlight bash %}
roscd package_name
# roscd roscpp
{% endhighlight %}

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> rosls

{% highlight bash %}
rosls package_name
# rosls roscpp
{% endhighlight %}


#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> Creating Package

catkin package는 다음의 조건을 만족시켜야 합니다.

* [catkin package.xml][ref-package.xml] 와 compliant해야 합니다.
* 패키지는 CMakeLists.txt 파일을 갖고 있어야 합니다.
* 같은 이름의 nested package는 허용되지 않습니다.

{% highlight bash %}
my_package/
  CMakeLists.txt
  package.xml
{% endhighlight %}

catkin workspace는 다음과 같은 모양이지만, 여기에서 src만 있는 standalone (build, devel, install제외)도 가능합니다.

{% highlight bash %}
workspace_folder/         -- WORKSPACE
  src/                    -- SOURCE SPACE
    CMakeLists.txt        -- The 'toplevel' CMake file
    package_1/
      CMakeLists.txt
      package.xml
      ...
    package_n/
      CMakeLists.txt
      package.xml
      ...
  build/                  -- BUILD SPACE
    CATKIN_IGNORE         -- Keeps catkin from walking this directory
  devel/                  -- DEVELOPMENT SPACE (set by CATKIN_DEVEL_PREFIX)
    bin/
    etc/
    include/
    lib/
    share/
    .catkin
    env.bash
    setup.bash
    setup.sh
    ...
  install/                -- INSTALL SPACE (set by CMAKE_INSTALL_PREFIX)
    bin/
    etc/
    include/
    lib/
    share/
    .catkin             
    env.bash
    setup.bash
    setup.sh
    ...
{% endhighlight %}

src 디렉토리로 이동한 다음에 **catkin_create_pkg** 를 이용해서 package 를 만듭니다.

{% highlight bash %}
cd src
catkin_create_pkg <package_name> [depend1] [depend2] [depend3]
# catkin_create_pkg beginner_tutorials std_msgs rospy roscpp
{% endhighlight %}

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> Building Packages

workspace의 root디렉토리에서 catkin_make로 만들어진 packages들을 build합니다.

{% highlight bash %}
# In a catkin workspace
source devel/setup.bash 
catkin_make
catkin_make install  # (optionally)
{% endhighlight %}

## <span class="glyphicon glyphicon-ok-sign" aria-hidden="true"></span> Nodes

많은 기기들, 센서들이 모여서 로봇이라는 하나의 큰 시스템을 이루게 됩니다. 즉 각각의 단위들을(기기들) 컨트롤하기 위해서 
Nodes라는 개념이 있으며, Nodes의 각각의 API에 따른 통신으로 서로 communicate할 수 있습니다.

각각의 노드들은 namespace를 갖게 되는데 Java또는 Android 의 package처럼 unique한 이름들을 갖게 됩니다.<br>
(See [Graph Resource Names][ref-names])

Node는 단순한 excutable file이 아니라, Node는 ROS Client Library를 통해서 Topic에 subscribe/publish 할 수 있습니다.
뿐만 아니라 Service를 제공/사용 할 수 있습니다. (rospy for Python, roscpp for C++)

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> roscore

먼저 노드들간에 통신을 하기 위해서는 roscore 서버를 켜야합니다. 

{% highlight bash %}
roscore
{% endhighlight %}

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> rosnode

rosnode사용시 반드시 source ./devel/setup.bash 를 해준다음에 환경변수가 설정된 이후에 사용해야합니다.<br>
list 명령어는 active nodes의 리스트만 출력시킵니다. 즉 현재 실행되고 있는 nodes만 출력합니다.

{% highlight bash %}
# Example
>rosnode list
/rosout

>rosnode info /rosout 
{% endhighlight %}

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> rosrun

rosrun은 package의 경로를 알필요 없이, 바로 package안의 a node를 실행시킬수 있도록 해줍니다.

{% highlight bash %}
rosrun [package_name] [node_name]
# rosrun turtlesim turtlesin_node
{% endhighlight %}


## <span class="glyphicon glyphicon-ok-sign" aria-hidden="true"></span> Topics

마치 Redis에서 Pub/Sub하듯이, ROS에서도 Nodes들간에 Pub/Sub하기 위한 채널로 Topics가 있습니다.<br>
rqt_graph를 사용하면 Nodes들간의 그래프를 그려줍니다.

{% highlight bash %}
sudo apt-get install ros-<distro>-rqt
sudo apt-get install ros-<distro>-rqt-common-plugins
rosrun rqt_graph rqt_graph
{% endhighlight %}

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> rostopic & rosmsg

**rostopic list**를 통해서 topics 리스트를 뽑을수 있고, **rostopic echo [topic]**을 통해서 메세지를 볼수 있습니다. <br>
**rostopic list -v** 명령어를 통해서 publicsher/subscriber 들을  볼 수 있습니다.

{% highlight bash %}
rostopic list -v
{% endhighlight %}

Nodes들간의 통신은 마치 Protobuf에서 메세지를 정의하듯이 동일한 **Type**의 메세지 교환으로 통신할수 있습니다.


{% highlight bash %}
rostopic type [topic name]
# rostopic type /turtle1/cmd_vel
# geometry_msgs/Twist

rosmsg show [message type]
# rosmsg show geometry_msgs/Twist
# geometry_msgs/Vector3 linear
#   float64 x
#   float64 y
#   float64 z
# geometry_msgs/Vector3 angular
#   float64 x
#   float64 y
#   float64 z
{% endhighlight %}

Bash Shell 에서 pub도 가능합니다.

{% highlight bash %}
rostopic pub [topic] [msg_type] [args]
# rostopic pub -1 /turtle1/cmd_vel geometry_msgs/Twist -- '[2.0, 0.0, 0.0]' '[0.0, 0.0, 1.8]'
{% endhighlight %}


## <span class="glyphicon glyphicon-ok-sign" aria-hidden="true"></span> Services

ROS Services는 Ndoes들이 서로 Communication 할 수 있는 또 다른 방법입니다.<br>
서비스는 노드들이 **Request**, **Response**를 서로 보내고 받을 수 있도록 해줍니다. 


#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> list

모든 ROS Commands들이 그렇듯이 list로 service list를 뽑을수 있습니다.
{% highlight bash %}
rosservice list
{% endhighlight %}


#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> type

{% highlight bash %}
rosservice type [service]
# >rosservice type clear
# std_srvs/Empty
{% endhighlight %}

Empty의 뜻은 request 그리고 response를 주고 받을때 데이터없이 주고 받는다는 뜻입니다.

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> call

{% highlight bash %}
rosservice call [service] [args]
# >rosservice call /clear
{% endhighlight %}

clear를 날리면 거북이 tutorial이 초기화됩니다. (거북이 위치, 지나온 path.. 등등 모두 초기화)<br>
arguments가 필요한 경우는 다음과 같습니다.

{% highlight bash %}
rosservice type spawn | rossrv show
# float32 x
# float32 y
# float32 theta
# string name
# ---
# string name
rosservice call spawn 2 2 0.2 "Hello"
{% endhighlight %}

spawn call을 날리면 해당 위치에 새로운 거북이가 만들어지게 됩니다.

## <span class="glyphicon glyphicon-ok-sign" aria-hidden="true"></span> Parameters

rosparam 명령어로 ROS Parameter Server의 데이터를 저장하거나, 수정하는일을 할 수 있습니다.<br>
Parameter Server는 마치 안드로이드의 Shared Preference 처럼 Node의 설정사항들을 저장할수 있습니다.<br>
즉 Binary데이터를 저장하기에는 적합하지 않으며, High Performance도 아닙니다.

{% highlight bash %}
rosparam set /helloworld 'hello world'
rosparam get /helloworld
hello world
{% endhighlight %}


dump 그리고 load로 전체 parameters data 를 파일로 저장또는 불러올수 있습니다.

{% highlight bash %}
rosparam dump [file_name] [namespace]
rosparam load [file_name] [namespace]
# rosparam dump param.yaml
{% endhighlight %}


## <span class="glyphicon glyphicon-ok-sign" aria-hidden="true"></span> Writing Publisher/Subscriber with Python

{% highlight bash %}
rospack list | grep begi
# beginner_tutorials /home/anderson/@ros/test/src/beginner_tutorials
roscd beginner_tutorials
mkdir scripts
cd scripts
vi talker.py
{% endhighlight %}

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> Publisher

먼저 **scripts** 디렉토리를 만들고 talker.py를 만듭니다.

{% highlight python %}
import rospy
from std_msgs.msg import String

def talker():
    # 해당 노드가 'chatter' 토픽으로 publish합니다.
    # 메세지 Type은 String 입니다.
    pub = rospy.Publisher('chatter', String, queue_size=10)
    
    # 'talker'라는 노드이름으로 초기화 
    # 이름은 base name이어야 하며, "/" slash가 있어서는 안됩니다. 
    rospy.init_node('talker', anonymous=True)
    
    # 초당 10번
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
{% endhighlight %}

* rospy.init_node('talker', anonymous=True) 시에 이름에 '/' slash 가 있어서는 안됩니다.

#### <span class="glyphicon glyphicon-ok-circle" aria-hidden="true"></span> Subscriber

{% highlight python %}
import rospy
from std_msgs.msg import String

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'talker' node so that multiple talkers can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("chatter", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
{% endhighlight %}

## <span class="glyphicon glyphicon-ok-sign" aria-hidden="true"></span> rviz

{% highlight python %}
rosrun rviz rviz
{% endhighlight %}

<img src="{{ page.asset_path }}rviz01.png" class="img-responsive img-rounded">

[ref-lily]: https://www.lily.camera/
[ref-catin]: http://wiki.ros.org/catkin/workspaces
[ref-package]: http://wiki.ros.org/ROS/Tutorials/NavigatingTheFilesystem
[ref-package.xml]: http://wiki.ros.org/catkin/package.xml
[ref-names]: http://wiki.ros.org/Names