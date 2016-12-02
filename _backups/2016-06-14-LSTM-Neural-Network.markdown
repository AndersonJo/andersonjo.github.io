---
layout: post
title:  "TensorFlow - LSTM Neural Network"
date:   2016-06-14 01:00:00
categories: "machine-learning"
asset_path: /assets/posts/LSTM-Neural-Network/
tags: ['Recurrent Neural Network', 'RNN', 'Long Short Term Memory NN']

---

<header>
    <img src="{{ page.asset_path }}wallpaper-weird-self-esteem.jpg" class="img-responsive img-rounded" style="width:100%">
</header>



# The Problem of RNN

RNN의 문제는 가장 최신의 이전 데이터를 통해서 현재를 볼 수 있지만.. 
아주 오래전 데이터 또는 Context를 통해서 현재를 보기 힘들다는 것입니다.

| O | The clouds are in the **sky** | context없이 sky를 추측할수 있습니다. |
| X | I grew up in France... I speak fluent **French** | 가장 최신 정보로는 어떤 language가 나오는 것은 알 수 있지만, French를 정확하게 알기 위해서는 France라는 Context를 알아야 합니다. |
 
즉 문제는, Gap이 커지면 커질수록 RNN은 정보를 연결시키지 못하는 단점을 갖고 있습니다. 
LSTM은 이 문제를 해결합니다.

# LSTM Networks

Long Short Term Memory Networks (LSTMs)는 long-term dependencies를 학습할수 있는 특수한 뉴럴 네크워크입니다.
일반적인 RNN은 다음과 같은 single tanh layer를 갖고 있는 형태를 띄고 있습니다.

<img src="{{ page.asset_path }}LSTM3-SimpleRNN.png" class="img-responsive img-rounded">

LSTM의 경우는 chain 같은 구조를 동일하게 갖고 있지만, 다른 repeating module 구조를 갖고 있습니다.

<img src="{{ page.asset_path }}LSTM3-chain.png" class="img-responsive img-rounded">

<img src="{{ page.asset_path }}LSTM2-notation.png" class="img-responsive img-rounded">

# The Core Idea Behind LSTMs

LSTM에서 중요한 부분은, cell state입니다. (아래의 다이어그램에서 가장 윗쪽에서 horizontal line을 가르킵니다.)<br>
cell state는 일종의 conveyer belt입니다. 

<img src="{{ page.asset_path }}LSTM3-C-line.png" class="img-responsive img-rounded">

LSTM은 어떤 정보를 cell state에 추가하거나 삭제를 할수 있으며, 이러한 것들은 gates에 의해서 조정이 됩니다.<br>
Gates는 sigmoid neural net layer 그리고 pointwise multiplication operation (핑크색 원) 으로 구성되어 있습니다.

<img src="{{ page.asset_path }}LSTM3-gate.png" class="img-responsive img-rounded text-center">

Sigmoid layer는 0 그리고 1사이의 값을 내며, 0값으로 나올수록 let nothing through의 의미이며, 
1값에 가까워질수록 let everything through! 라는 의미입니다.
LSTM은 cell state를 보호하거나 컨트롤하기위해 이런 gates를 3개 갖고 있습니다.

# LSTM Walk Through


LSTM의 첫번째 단계는 어떤 정보를 cell state에서 내보낼것인가 입니다.
이러한 결정은(내보내는) "forget gate layer"라는 sigmoid layer에서 결정이 됩니다.
아래 그림과 같은 sigmoid layer거치고 나서 output은 0~1값이며, 0은 해당 cell state 정보를 삭제시키는 것이며, 
1값은 유지시키는 것입니다. (keep this!) 

예를 들어서, 이전의 단어들에 기반하여 다음 단어를 예측할때, 현재 주어의 성별(he, she, you)을 cell state에 포함시킴으로서 
정확한 pronouns(him, her, me)가 사용되도록 할 수 있습니다. 그리고 새로운 subject (주어)를 찾게되면, 이전의 gender의 정보는 
cell state에서 삭제시키도록 만들면 됩니다.

<img src="{{ page.asset_path }}LSTM3-focus-f.png" class="img-responsive img-rounded text-center">

그 다음 단계는, 어떤 새로운 정보를 cell state에 저장시킬지 입니다. 이 부분은 2가지 부분으로 나뉩니다.<br>
첫번째는 **"input gate layer"**라고 불리는 sigmoid layer이며, 어떤 값(values)를 업데이트해줄지 결정합니다.<br>
두번째는 tanh layer가 a vector of new candidate values, $$ \tilde{C}_t $$ 를 생성하며 state에 포함이 될 수도 있습니다.<br>
그 다음으로 이 두개의 값을 합쳐서 state를 업데이트 시켜줍니다.

<img src="{{ page.asset_path }}LSTM3-focus-i.png" class="img-responsive img-rounded text-center">

이제 old cell state, $$ C_{t-1} $$ 를 $$ C_{t} $$ 로 업데이트 해줄 차례 입니다. <br>
old state (사라질 정보) 를 $$ f_t $$로 multiply해줍니다. 그런 다음 $$ i_t*\tilde{C}_t $$(new candidate values)을 더해줍니다.<br>
즉 예제에서는, 실질적으로 old subject's gender(he, she 등등)에 대한 정보를 삭제시키고, 새로운 정보를 추가시키는 작업입니다.

<img src="{{ page.asset_path }}LSTM3-focus-C.png" class="img-responsive img-rounded text-center">

마지막으로 어떤 값을 output으로 내놓을지 결정합니다. 즉 filtered version 의 cell state이겠죠.<br>
첫번째로 일단 sigmoid layer에 돌려서 cell state의 어떤 부분을 output으로 내놓을지를 정합니다. 
그런다음 그 cell state를 tanh에 통과시킵니다. (값들을 -1 ~ 1사이로 만듭니다.) 그리고 sigmoid gate를 나온 output값과 곱해줍니다.
즉 통과시킨 일부분만 output으로 뽑히겠죠. 

<img src="{{ page.asset_path }}LSTM3-focus-o.png" class="img-responsive img-rounded text-center">

# Variants on Long Short Term Memory

위의 LSTM은 normal한 경우이고, 여러가지 버젼의 LSTM이 존재합니다. 
실제로 거의 모든 문서들이 각기 다른 버젼의 LSTM을 사용합니다. 
그중에 유명한  LSTM variant는 [Gers & Schmidhuber (2000)][Gers & Schmidhuber (2000)] 에 의해서 소개된 논문이며, "peephole connections"을 추가했습니다.

<img src="{{ page.asset_path }}LSTM3-var-peepholes.png" class="img-responsive img-rounded text-center">

다른 LSTM variant는 coupled forget and input gates를 사용합니다. 즉 무엇을 잊을지 그리고 무엇을 추가할지를 따로 구분짓는것이 아니라 한곳에서 한번에 결정을 내립니다.
해당 variant는 old state를 잊을때에 새로운 state를 넣거나, 새로운 정보를 넣어야할때 old state를 잊게 됩니다.

<img src="{{ page.asset_path }}LSTM3-var-tied.png" class="img-responsive img-rounded text-center">

다른 variation on LSTM은 [Cho, et al. (2014)][Cho, et al. (2014)]의해 소개된 Gated Recurrent Unit, or GRU입니다.
해당 LSTM은 forget 과 input gates를 하나의 update gate로 통합시킨것입니다.
또한 cell state와 hidden state를 합쳤습니다. 결과적으로 standard LSTM모델보다 더 심플하며, 매우 유명합니다.

<img src="{{ page.asset_path }}LSTM3-var-GRU.png" class="img-responsive img-rounded text-center">

# References or Almost Translations

**위의 문서는 [Colah's Blog][Colah's Blog]에서 거의 번역수준으로 가져왔습니다. (그냥 제가 이해하기 쉬운 말로 풀어쓴 글입니다.) 자세한 정보를 써준 Colar님? 에게 무한 감사를 표합니다.**



[Colah's Blog]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[Gers & Schmidhuber (2000)]: ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf
[Cho, et al. (2014)]: http://arxiv.org/pdf/1406.1078v3.pdf