---
layout: post
title:  "Neural Network for Concrete Strength using R"
date:   2015-07-25 15:00:00
categories: "neural-network"
asset_path: /assets/posts/Neural-Network-for-concrete/
---
<div>
    <img src="{{ page.asset_path }}concrete.jpg">
</div>
신경망이라는 주제는 Machine Learning분야에서 80년대부터 계속해서 발전해온 기술입니다.<br>
맛보기로 R을 사용해서 신경망(Neural Network)을 해보도록 하겠습니다.<br>

콘크리트의 퍼포먼스 즉 강도(Strength)를 결정하는데에는 많은 변수들이 있습니다.<br>
들어가는 재료들의 양에 따라서 콘크리트의 강도가 달라지기 때문에 여기에는 특별한 수학적 공식을 얻어내기가 쉽지가 않습니다. <br>

Neural Network를 이용하면 특별한 수학적 공식 (콘크리트의 강도를 알아내는..) 없이 ANN(Artificial Neural Network)을 트레이닝 시키고
훈련된 ANN으로 다시 새로운 데이터로 예측을 할 것입니다.<br> 

<h3>Data & References</h3>
[concrete.csv][csv]<br>
[example.R][r]<br>
[Machine Learning with R][book] 

<h3>Neuralnet 설치 </h3>

여러 Neural Network 라이브러리들이 있지만.. Neuralnet이라는 라이브러리를 사용할 것입니다.

{% highlight r %}
install.packages("neuralnet")
{% endhighlight %}

<h3>일단 코드로 써보자</h3>

read.csv 함수를 통해서 concrete.csv파일을 읽습니다.<br>
stringAsFactors는 R의 데이터타입중에 string을 factor로 읽지 않고  vector로 읽겠다는 뜻이며, F는 FALSE(boolean)값과 같습니다.<br>
끝에 [-1]은 첫번째 row 필드값을 exclude시키겠다는 말..<br><br>

포인트는.. cement, flag, ash, water, superplastic, coarseagg, findagg, age 등등 다양한 요소가 모여서 궁극적으로 strength를 
결정을 하게 됩니다. 각각의 재료및 요소에 따라서 strength의 값이 달라지는데.. 역시 정확한 공식을 얻기는 쉽지 않겠죠..

{% highlight r %}
concrete <- read.csv('concrete.csv', stringsAsFactors = F)[-1]
head(concrete)

  cement  flag ash water superplastic coarseagg findagg age strength
1  540.0   0.0   0   162          2.5    1040.0   676.0  28    79.99
2  540.0   0.0   0   162          2.5    1055.0   676.0  28    61.89
3  332.5 142.5   0   228          0.0     932.0   594.0 270    40.27
4  332.5 142.5   0   228          0.0     932.0   594.0 365    41.05
5  198.6 132.4   0   192          0.0     978.4   825.5 360    44.30
6  266.0 114.0   0   228          0.0     932.0   670.0  90    47.03

{% endhighlight %}

데이터를 살펴보니.. 각각의 필드들이 제각각의 범위의 값을 갖고 있습니다. 즉.. Normalize가 필요합니다.<br>
여러방법의 normalize가 있는데 우리는 min max normalize를 사용하겠습니다. 

{% highlight r %}

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
concrete <- as.data.frame(lapply(concrete, normalize))
head(concrete)

        cement         flag ash        water  superplastic    coarseagg      findagg           age     strength
1 1.0000000000 0.0000000000   0 0.3210862620 0.07763975155 0.6947674419 0.2057200201 0.07417582418 0.9674847390
2 1.0000000000 0.0000000000   0 0.3210862620 0.07763975155 0.7383720930 0.2057200201 0.07417582418 0.7419957643
3 0.5262557078 0.3964941569   0 0.8482428115 0.00000000000 0.3808139535 0.0000000000 0.73901098901 0.4726547901
4 0.5262557078 0.3964941569   0 0.8482428115 0.00000000000 0.3808139535 0.0000000000 1.00000000000 0.4823719945
5 0.2205479452 0.3683917641   0 0.5607028754 0.00000000000 0.5156976744 0.5807827396 0.98626373626 0.5228603463
6 0.3744292237 0.3171953255   0 0.8482428115 0.00000000000 0.3808139535 0.1906673357 0.24450549451 0.5568705619

{% endhighlight %}

lapply함수에 바로 dataframe type을 넣을수가 있고 lapply는 각각의 필드마다 normalize함수를 진행하게 됩니다.<br>
가장 큰 값은 1이 되었고.. 가장 작은 값은 0으로 normalize를 하였습니다.<br>

<blockquote>
만약 lapply를 사용하지 않고 바로 normalize(concrete) 를 사용하게 되면.. dataframe내에서 가장 큰 값이 1을 갖고, 가장 작은 값이 0을
갖게 됩니다. 즉 cement필드에서 0~1 또는.. water에서 0~1값을 갖는게 아니라.. 전체 dataframe내에서 normalize를 하게 되며.. 
이는 우리가 원하는 결과값이 아닙니다.
</blockquote>

그 다음으로.. concrete 데이터의 75%를 트레이닝용으로 사용하고, 나머지는 해당 neural network가 제대로 돌아가는지 검증용으로 사용을 할 
것입니다. 

{% highlight r %}

max_index <- length(concrete$strength)
slice_index <- round(max_index * 0.75)
training_data <- concrete[1:slice_index,]
test_data <- concrete[slice_index:max_index,]

{% endhighlight %}

<h3>Neural Network 트레이닝 시키기</h3>
자 이제 모든 준비가 끝났으니 Neural Network를 트레이닝 시키겠습니다. <br>

{% highlight r %}

concrete_model <-
  neuralnet(strength ~ cement + flag + ash + water + superplastic + coarseagg +
      findagg + age, data = training_data, hidden = 3
  )
plot(concrete_model) # plot the neural networks

{% endhighlight %}

<img src="{{page.asset_path}}neuralnet_hidden_3.png">

위의 그림에 보면 8개의 nodes가 보입니다. 이것을 Input Nodes라고 부릅니다.<br>
중간에 3개의 원이 있는데 이것을 Hidden Nodes라고 부릅니다.<br>
마지막 제일 오른쪽에 1개의 원이있는데 이것을 Output Node라고 부릅니다.<br>

Backpropagation이 있는데.. 트레이닝중에 틀리거나 맞는것이 있을때마다 결과치에 따라서 각각의 Nodes들의 수치를 변경하게 됩니다.<br>
즉 각각의 Nodes들에는 서로 연결되어있어서 강하게 연결되어 있는것이 있고, 약하게 연결된것이 있는데.. 
Backpropagation을 통해서 이 가중치를 변경을 하게 되는 것입니다.<br>

아! 참고로 최초로 Nodes들이 연결이 될때는 어떻게 가중치를 주는가? 인데..
최초에는 랜덤값을 그냥 넣습니다. 따라서 여러분이 집접 저 코드를 돌려보면 매번 다른 값이 나오게 될 것입니다.<br>

자! 그렇다면 이렇게 학습된 Neural Network의 성능시험이 필요하지 않을까요?
처음에 75%데이터를 학습으로 사용했으니, 나머지 데이터를 이용해서 정확하게 예측을 하는지 검사해봅시다.<br>

<h3> Neural Network 성능측정 (Prediction) </h3>

성능측정은 상관관계분석(Correlation)을 통해서 측정을 하게 됩니다.<br>
1값이 나올수록 예측의 정확성이 높다는 뜻이고, 0값이 나온다면 예측값이 떨어진다는 뜻입니다. 

{% highlight r %}
model_results <- compute(concrete_model, test_data[1:8])
cor(model_results$net.result, test_data$strength)
0.8201494476
{% endhighlight %}






[bitbucket]: https://jochangmin@bitbucket.org/jochangmin/r-examples.git
[book]: http://www.amazon.com/Machine-Learning-R-Brett-Lantz/dp/1782162143/ref=sr_1_1?ie=UTF8&qid=1437813079&sr=8-1&keywords=machine+learning+with+r
[csv]: {{page.asset_path}}concrete.csv
[r]: {{page.asset_path}}example.R




[anderson-linkedin]: https://kr.linkedin.com/in/anderdson
[email]: a141890@gmail.com
