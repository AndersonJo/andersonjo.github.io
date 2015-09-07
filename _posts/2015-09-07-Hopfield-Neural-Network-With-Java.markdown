---
layout: post
title:  "Hopfield Neural Network with Java"
date:   2015-09-07 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/Hopfield-Neural-Network/
---
<div>
    <img src="{{ page.asset_path }}robot.jpg" class="img-responsive img-rounded">
</div>

## Preliminary

[Github Code][github-ann]

gradle로 관련 라이브러리들을 쉽게 설치가능합니다.

{% highlight shell %}

gradle build
gradle eclipse

{% endhighlight %}

## Matrix - Apache Math

Matrix 라이브러리로  Jama 그리고 Apache Common Math 도 있는데 필자는 Apache Common Math를 사용하도록 하겠습니다.
기본적으로 다음과 같은 값을 준비합니다.

{% highlight java %}
double[][] dataA = { { 1, 2, 3 }, { 0, 4, 5 } };
double[][] dataB = { { 3, 3, 3 }, { 3, 2, 0 } };

RealMatrix a = MatrixUtils.createRealMatrix(dataA);
RealMatrix b = MatrixUtils.createRealMatrix(dataB);

{% endhighlight %}

#### Addition

{% highlight java %}

a.add(b);
// { {4.0,5.0,6.0},
//   {3.0,6.0,5.0} }

{% endhighlight %}

#### Multiplication

먼저 a, b의 matrix의 dimension이 서로 맞지 않으니 transpose로 row와 columns을 서로 바꾸도록 합니다.

{% highlight java %}

b  = b.transpose();
b;
// { {3.0,3.0},{3.0,2.0},{3.0,0.0} }

a.multiply(b); // Dot Multiplication
// { {18.0,7.0},{27.0,8.0} }

a.scalarMultiply(3);
// { {3.0,6.0,9.0},{0.0,12.0,15.0} }

a.preMultiply(b)
// { { 3.0, 18.0, 24.0 }, { 3.0, 14.0, 19.0 }, { 3.0, 6.0, 9.0 } }


{% endhighlight %}

#### Identity Matrix

1 * x = x 처럼 자기 자신을 나오게 하는 1을 identity 라고 합니다.
마찬가지로 Identity Matrix가 있는데 row, column이 동일하고 대각선으로 1이 있는 모습니다.

{% highlight java %}

double[][] dataA = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };

RealMatrix a = MatrixUtils.createRealMatrix(dataA);
RealMatrix b = MatrixUtils.createRealIdentityMatrix(3);

System.out.println(ReflectionToStringBuilder.toString(b));
// { {1.0,0.0,0.0},
//   {0.0,1.0,0.0},
//   {0.0,0.0,1.0} }

a.multiply(b);
// { {1.0,2.0,3.0},
//   {4.0,5.0,6.0},
//   {7.0,8.0,9.0} }

{% endhighlight %}

## Bipolar Notation

Binary에서는 1은 True를 나타내고, 0은 False를 나타냅니다.<br>
Bipolar Notation에서는 1은 True를 나타내고 -1이 False를 나타냅니다.
 
때문에 서로 변환이 필요할때가 있습니다. 

**Binary --> Bipolar**

<img src="{{ page.asset_path }}to_bipolar.jpg" class="img-responsive img-rounded">

**Bipolar --> Binary**

<img src="{{ page.asset_path }}to_binary.jpg" class="img-responsive img-rounded">

예를 들어서 Binary 0은 2*0-1 = -1 이 되고, 
Bipolar -1 은 (-1 + 1)/2= 0 이 됩니다.




## Hopfield Neural Network

<img src="{{ page.asset_path }}hopfield.jpg" class="img-responsive img-rounded">

Hopfiled Neural Network는 가장 단순한 ANN중의 하나입니다. 
일단 Single Layer이며, Autoassociative Network입니다.
Single Layer라는 뜻은 각각의 모든 neurons 들이 모두 연결이 되어 있으며, 
Autoassociative 라는 뜻은 어떤 한 패턴을 인지하면 해당 패턴을 리턴시킨다는 뜻입니다. 



#### Initialization
먼저 contribution matrix 를 만듭니다. 여기에서는 [0, 1, 0, 1] 로 만들도록 하겠습니다. 

{% highlight java %}
double[] data = { 0, 1, 0, 1 };
double[] data2 = new double[4];
double[] data3 = {1, 0, 0, 1};
for (int i = 0; i < data.length; i++) {
    data2[i] = 2 * data[i] - 1;
}

RealMatrix inputPatternMatrix = MatrixUtils.createRealMatrix(1, 4);
inputPatternMatrix.setRow(0, data); // {{0, 1, 0, 1}}

RealMatrix contributionMatrix = MatrixUtils.createRowRealMatrix(data2); 
// { {-1},{1},{-1},{1} }
{% endhighlight %}

#### Connection

contributionMatrix가 만들어졌으면 자기 자신을 multiplication해줍니다. 
즉 모든 neural들을 연결시켜주는것과 같습니다.

{% highlight java %}
contributionMatrix = contributionMatrix.preMultiply(contributionMatrix.transpose());
//{ {1.0,-1.0,1.0,-1.0},
//  {-1.0,1.0,-1.0,1.0},
//  {1.0,-1.0,1.0,-1.0},
//  {-1.0,1.0,-1.0,1.0} }
{% endhighlight %}

이렇게 나온 결과물에 다시 Identity Matrix를 subtract해줍니다.
이렇게 해주는 이유는 자기 자신의 neuron과는 연결이 되어 있지 않기 때문입니다.

{% highlight java %}
RealMatrix identityMatrix = MatrixUtils.createRealIdentityMatrix(4);
contributionMatrix = contributionMatrix.subtract(identityMatrix);
//{ {0.0,-1.0,1.0,-1.0},
//  {-1.0,0.0,-1.0,1.0},
//  {1.0,-1.0,0.0,-1.0},
//  {-1.0,1.0,-1.0,0.0} }
{% endhighlight %}

#### Verification

이렇게 Hopfiled Neural Network에 대한 트레이닝이 끝났고 실제 input 값을 넣었을때 
동일한 값이 나오는지 확인이 필요합니다. 

{% highlight java %}
RealMatrix result = contributionMatrix.multiply(inputPatternMatrix.transpose());
HopfieldTutorial.filterOne(result);
System.out.println(ReflectionToStringBuilder.toString(result.transpose()));
// { {0, 1, 0, 1} }

{% endhighlight %}

동일한 0, 1, 0, 1이 나오는 것을 확인할수 있습니다. 여기서 filterOne은 1이 아니면 0으로 값을 변경해주는 함수 입니다.

{% highlight java %}
RealMatrix wrongMatrix = MatrixUtils.createColumnRealMatrix(data3);
// { {1},{0},{0},{1} }
result = contributionMatrix.multiply(wrongMatrix);
HopfieldTutorial.filterOne(result);
// { {0},{0},{0},{0} }

{% endhighlight %}

[github-ann]: https://github.com/AndersonJo/Neural-Network-Tutorial