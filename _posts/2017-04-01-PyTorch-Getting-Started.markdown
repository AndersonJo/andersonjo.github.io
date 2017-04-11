---
layout: post
title:  "Pytorch Getting Started"
date:   2017-04-01 01:00:00
categories: "machine-learning"
asset_path: /assets/posts2/Pytorch/
tags: ['Pytorch']

---

<header>
    <img src="{{ page.asset_path }}logo.jpg" class="img-responsive img-rounded" style="width:100%">
    <div style="text-align:right;"> 
    <small>Keras로 GAN만들다가 trainable = False 로 (Layer까지 다 줬음)해도 training 되서 **개빡치고** Pytorch시작함.<br>
    Keras에서는 compile전에 trainable먹히고, compile이후에는 dynamic하게 바뀌지 않음.  18
    </small>
    </div>
</header>

# Installation

### Install Dependencies

{% highlight bash %}
sudo apt-get install cmake
sudo pip3 install cffi
{% endhighlight %}

### Install Pytorch from Source

{% highlight bash %}
git clone https://github.com/pytorch/pytorch.git
cd pytorch

sudo python3.6 setup.py install
{% endhighlight %}

### Install Torchvision from source

{% highlight bash %}
git clone https://github.com/pytorch/vision.git
cd vision
sudo python3.6 setup.py install
{% endhighlight %}

# Useful Tips

### Data Types

| Data type	| CPU tensor | GPU tensor | Numpy Type | Numpy Type Description |
|:----------|:-----------|:-----------|:-----------|:-----------------------|
| 32-bit floating point | torch.FloatTensor | torch.cuda.FloatTensor | float32 |  Single precision float: sign bit, 8 bits exponent, 23 bits mantissa |
| 64-bit floating point | torch.DoubleTensor | torch.cuda.DoubleTensor | float64 | Double precision float: sign bit, 11 bits exponent, 52 bits mantissa |
| 16-bit floating point | N/A | torch.cuda.HalfTensor | float16 | Half precision float: sign bit, 5 bits exponent, 10 bits mantissa |
| 8-bit integer (unsigned) | torch.ByteTensor | torch.cuda.ByteTensor | uint8 | Unsigned integer (0 to 255) |
| 8-bit integer (signed) | torch.CharTensor | torch.cuda.CharTensor | int8 | Byte (-128 to 127) |
| 16-bit integer (signed) | torch.ShortTensor | torch.cuda.ShortTensor | int16 | Integer (-32768 to 32767) |
| 32-bit integer (signed) | torch.IntTensor | torch.cuda.IntTensor | int32 | Integer (-2147483648 to 2147483647) |
| 64-bit integer (signed) | torch.LongTensor | torch.cuda.LongTensor | int64 | Integer (-9223372036854775808 to 9223372036854775807) |


# Tutorial

### Operate on GPU

Tensor들은 cuda함수를 통해서 GPU로 이동시킬수 있습니다.

{% highlight python %}
a = torch.rand(5, 3).cuda()
b = torch.rand(3, 5).cuda()
a.dot(b) # 4.939548492431641
{% endhighlight %}

### Autograd

#### Variable

autograd.Variable class는 tensor를 wrapping하고 있으며, 대부분의 연산은 Variable을 통해서 이루어지게 됩니다.<br>
연산을 마친후, .backward()함수를 통해서 자동으로 gradients를 구할 수 있습니다.

{% highlight python %}
>> x = Variable(torch.ones(4, 4), requires_grad=True)
>> x.data

 1  1  1  1
 1  1  1  1
 1  1  1  1
 1  1  1  1
[torch.FloatTensor of size 4x4]
{% endhighlight %}

**.creator**는 해당 Variable을 만든 Function을 reference합니다. <br>

{% highlight python %}
>> y = x * 3
>> y.creator
<torch.autograd._functions.basic_ops.MulConstant at 0x7f33f0b85e48>
{% endhighlight %}

#### Gradients

.backward() 함수를 사용하여 computation에 대한 gradients값을 구할 수 있습니다.<br>
(아래의 공식에서 o는 output변수를 가르킵니다.)

$$ \begin{align}
z_i &= 3(x_i+2)^2 \\
o &= \frac{1}{4}\sum_i z_i
\end{align} $$

위 공식에 대한 gradient값을 구하면 다음과 같습니다.

$$ \begin{align}
\frac{\partial o}{\partial x_i} &= \frac{3}{2}(x_i+2) & [1] \\
\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} &= \frac{9}{2} = 4.5 & [2]
\end{align} $$

* [2]에서 $$ x_i = 1 $$ 이 주어졌을때, o의 x에 대한 gradient 값은 4.5 가 된다는 뜻입니다.

{% highlight python %}
>> x = Variable(torch.ones(2, 2), requires_grad=True)
>> y = x + 2
>> z = y**2 * 3
>> out = z.mean() # 27
>> out.backward()

>> x.grad
Variable containing:
 4.5000  4.5000
 4.5000  4.5000
[torch.FloatTensor of size 2x2]
{% endhighlight %}

