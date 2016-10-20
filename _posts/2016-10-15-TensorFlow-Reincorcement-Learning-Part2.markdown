---
layout: post
title:  "TensorFlow - Deep Reinforcement Learning Part 2"
date:   2016-10-15 01:00:00
categories: "tensorflow"
asset_path: /assets/posts2/TensorFlow/
tags: ['OpenAI', 'Neon', 'format']

---

<header>
    <img src="{{ page.asset_path }}google-breakout.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small>구글 이미지에서 Atari breakout이라고 치면 게임이 나옴. ㅎㄷㄷㄷㄷ        
    </small>
    </div>
</header>

# Installation

### Environment

GPU를 설치하기 위해서는 다음의 environment variable이 추가되어 있어야 합니다.<br>
PyCUDA 설치시 cuda.h를 못찾아서 에러나는데 sudo실행시키면서 root 계정에서는 cuda를 못찾아서 발생하는 것.<br>
sudo su 누르고 .bashrc 에 추가시켜주면 됨. (nvcc 도 확인해볼것)

{% highlight bash %}
export PATH=$PATH:/usr/local/cuda/bin:/usr/local/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/lib
{% endhighlight %}

### Installing Arcade Learning Environment

[The Arcade Learning Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment) AI Research에 사용되며, Stella, Atari 2600 VCS Emulator를 기반으로 하고 있습니다.

**prerequisites**

{% highlight bash %}
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
{% endhighlight %}

**Install ALE**

{% highlight bash %}
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
make -j 4
sudo pip install .
{% endhighlight %}


### Installing PyCUDA (for supporting Neon GPU)

어려울것 없지만, root계정에 위의 environment variables이 존재해야 합니다.<br>
Neon설치시 자동으로 설치되지만, 에러가 많이 일으켜서 따로 먼저 설치해주고 Neon을 설치하는게 좋습니다.

{% highlight bash %}
sudo pip install pycuda
{% endhighlight %}

### Installing Neon

자세한 내용은 [Neon 설치 문서](http://neon.nervanasys.com/docs/latest/installation.html)를 참고하시기 바랍니다.<br>
Prerequisites을 설치합니다.<br>


{% highlight bash %}
sudo apt-get install libhdf5-dev libyaml-dev libopencv-dev pkg-config
sudo apt-get install python python-dev python-pip python-virtualenv
{% endhighlight %}

Neon을 설치합니다.

{% highlight bash %}
git clone https://github.com/NervanaSystems/neon.git
cd neon
sudo make -e VIS=true sysinstall
{% endhighlight %}

### For Game Video (Optional)

게임 비디오 프로듀스하기위해서는 다음을 설치합니다. 

{% highlight bash %}
sudo apt-get install libav-tools
{% endhighlight %}

### Installing OpenAI Gym

{% highlight bash %}
sudo pip install gym[all]
{% endhighlight %}


# Deep Q Learning with TensorFlow

### Predicting Network

{% highlight python %}
states = tf.placeholder('float32', [None, self.history_length, self.screen_height, self.screen_width])

# Convolutional Neural Network
net = tflearn.conv_2d(self.states, 32, 8, strides=4, activation='relu')
net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
net = tflearn.conv_2d(net, 64, 3, strides=1, activation='relu')

# Deep Neural Network
net = fully_connected(net, 512, activation='relu', name='l4')
q_values = tflearn.fully_connected(net, self.env.action_size, name='q')
q_action = tf.argmax(q_values, dimension=1)
{% endhighlight %}


### Target Network

{% highlight python %}
target_states = tf.placeholder('float32', [None, self.history_length, self.screen_height, self.screen_width], name='target_s_t')
target_net = tflearn.conv_2d(self.target_states, 32, 8, strides=4, activation='relu')
target_net = tflearn.conv_2d(target_net, 64, 4, strides=2, activation='relu')
target_net = tflearn.conv_2d(target_net, 64, 3, strides=1, activation='relu')

target_net = fully_connected(target_net, 512, activation='relu', name='target_l4')
target_q = tflearn.fully_connected(target_net, self.env.action_size, name='target_q')

target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
target_q_with_idx = tf.gather_nd(target_q, self.target_q_idx)  # Matrix Indexing
{% endhighlight %}



### References

* [OpenAI GYM](https://gym.openai.com/)
* [Guest Post (Part II): Deep Reinforcement Learning with Neon](https://www.nervanasys.com/deep-reinforcement-learning-with-neon/)
* [Blog Post (Part III): Deep Reinforcement Learning with OpenAI Gym](https://www.nervanasys.com/openai/)
* [DQN-tensorflow]: https://github.com/devsisters/DQN-tensorflow