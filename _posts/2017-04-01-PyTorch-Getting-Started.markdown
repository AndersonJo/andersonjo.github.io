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
sudo apt-get install cmake xvfb libav-tools xorg-dev libsdl2-dev swig

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

# MNIST Tutorial


### Data

* [Pytorch Transform Documentation](http://pytorch.org/docs/torchvision/transforms.html)


1. **torchvision.transforms.Compose:** 여러개의 tranforms을 실행합니다.
2. **torchvision.transforms.ToTensor:** PIL.Image 또는 [0, 255] range의 Numpy array(H x W x C)를 (C x H x W)의 **[0.0, 1.0] range**를 갖은 torch.FloatTensor로 변형시킵니다. <br>여기서 포인트가 0에서 1사이의 값을 갖은 값으로 normalization이 포함되있습니다.
3. **dataloader.DataLoader:** 사용하여 training시킬때 1개의 batch를 가져올때 shape이 **torch.Size([64, 1, 28, 28])** 이렇게 나옵니다.

{% highlight python %}
train = MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(), # ToTensor does min-max normalization.
]), )

test = MNIST('./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(), # ToTensor does min-max normalization.
]), )

# Create DataLoader
dataloader_args = dict(shuffle=True, batch_size=64,num_workers=1, pin_memory=True)
train_loader = dataloader.DataLoader(train, **dataloader_args)
test_loader = dataloader.DataLoader(test, **dataloader_args)
{% endhighlight %}

### Model

{% highlight python %}
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(784, 548)
        self.bc1 = nn.BatchNorm1d(548)

        self.fc2 = nn.Linear(548, 252)
        self.bc2 = nn.BatchNorm1d(252)

        self.fc3 = nn.Linear(252, 10)


    def forward(self, x):
        x = x.view((-1, 784))
        h = self.fc1(x)
        h = self.bc1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        h = self.fc2(h)
        h = self.bc2(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        h = self.fc3(h)
        out = F.log_softmax(h)
        return out

model = Model()
model.cuda() # CUDA!
optimizer = optim.Adam(model.parameters(), lr=0.001)
{% endhighlight %}

### Train

{% highlight python %}
model.train()

losses = []
for epoch in range(15):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get Samples
        data, target = Variable(data.cuda()), Variable(target.cuda())

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = F.cross_entropy(y_pred, target)
        losses.append(loss.data[0])
        # Backpropagation
        loss.backward()
        optimizer.step()


        # Display
        if batch_idx % 100 == 1:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0]),
                end='')

    print()
{% endhighlight %}

## Evaluate

{% highlight python %}
evaluate_x = Variable(test_loader.dataset.test_data.type_as(torch.FloatTensor())).cuda()
evaluate_y = Variable(test_loader.dataset.test_labels).cuda()

output = model(evaluate_x)
pred = output.data.max(1)[1]
d = pred.eq(evaluate_y.data).cpu()
accuracy = d.sum()/d.size()[0]

print('Accuracy:', accuracy) # Accuracy: 0.9764
{% endhighlight %}