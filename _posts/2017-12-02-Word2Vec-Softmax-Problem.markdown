---
layout: post
title:  "Word2Vec - The problem of Softmax & Information Content"
date:   2017-11-25 01:00:00
categories: "nlp"
asset_path: /assets/images/
tags: ['word2vec', 'softmax', 'skipgram', 'sampling', 'nce', 'noise-constrastive-estimation', 'information-theory', 'entropy']
---

# Word2Vec

NLP분야에서 어떻게하면 단어를 효율적으로 vector로 만들어, 기계학습 알고리즘에 넣는가입니다. <br>
2000년부터 많은 기법과 시도가 있었지만, 최근 기계학습분야에서 가장 많이 쓰이는 방법은 "Word2Vec"라는 방법입니다.

단어를 vector화 시키는 가장 쉬운 방법은 one-hot encoding으로 변환시키는 방법입니다. <br>
하지만 one-hot vector의 문제점은 단어의 갯수가 50,000개가 넘을수도 있고 많은 양의 단어를 다뤄야 하는데..
그만큼 vector의 크기가 단어의 갯수만큼 늘어난 다는 것입니다.
vector의 크기가 커질수록 효율성은 매우 떨어지게 됩니다. 예를 들어 NN에서 해당 input을 받기 위해서는 최소 50,000개 이상의 노드를 필요로 하게 됩니다.

또한 one-hot vector의 문제점은 단어들관의 연관성도 없다는 것입니다. <br>
예를 들어서 "케익" 이라는 단어를 들으면 "카페", "맛있다", "커피", "생일" 과 같은 단어들과 연관성이 있을것입니다. <br>
하지만 one-hot vector의 문제점은 문장안에서의 이러한 연관관계에 대한 정보를 잃어버리게 됩니다.

Word embedding은 단어간의 의미를 포함하는 dense vector를 얻도록 합니다.

여러가지 모델중에서 현재까지 가장 많이 쓰이는 embedding model은 Word2Vec 입니다.<br>
2013년 Mikolov et al가 2개의 페이퍼를 내놓았습니다.

첫번째 페이퍼는  [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) 으로서, 기존 모델에 비해서 더 효율적인 computation을 갖은 2개의 word embedding architectures를 제안합니다.

두번째 페이퍼는 [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) 으로서, 첫번째 페이퍼에서 내놓은 2개의 모델을 개선함으로서, 속도 그리고 정확도를 더 높였습니다.




## Skip-Gram Model

아래는 orginal skip-gram 모델 아키텍쳐이며, 목표는 주변의 단어를 잘 예측하는 word vector representations 를 학습하는 것 입니다.<br>
학습할 일련의 단어들 $$ w_1, w_2, w_3, ..., w_t $$ 가 주어졌을때, Skip-gram 모델의 목표(objective)는 average log probability를 maximize하는 것입니다.<br>
$$ w_t $$ 는 중심이 되는 단어 이며, $$ c $$ 는 context의 크기이며, 해당 값이 높을수록 accuracy는 높아지지만, training time은 반비례적으로 늘어나게 됩니다.

$$ \begin{align}
J_{\theta} = \frac{1}{T} \sum^T_{t=1} \sum_{-c\ \le\ j\ \le\ c, \ j \ne 0} \log p \left(w_{t + j} | w_t \right)
\end{align} $$



자세한 그림을 갖고서 설명하면 다음과 같습니다.<br>
위의 공식에서 $$ w_t $$ 는 아래 그림에서 $$ x_k $$ 와 동일하며, $$ w_{t+j} $$ 는 아래 그림에서 $$ y_{c,j} $$ 와 동일합니다.



<img src="{{ page.asset_path }}word2vec_skipgram2.png" class="img-responsive img-rounded img-fluid">


예를 들어서 `커피` 라는 단어가 주어졌을때, `얼음` 이라는 단어가 나올 확률은.. P("얼음" | "커피") 을 알아보도록 하겠습니다. <br>
$$ x_k $$ 는 `커피` 라는 one-hot vector이며, $$ W $$ 는 V x N 형태의 weight matrix로서 word embedding에서 우리가 알아내고자 하는 matrix입니다. one-hot vector와 matrix를 곱하게 되면 실질적으로는 $$ W $$ 의 1 x N (N은 neurons의 갯수) vector를 그냥 꺼내오게 됩니다. 이후 output layer $$ W' $$  (N x V) 와 dot product연산을 하게 되며 결과적으로 C x V dimension의 vectors가 만들어지게 됩니다.







이후 softmax function을 사용하여 각 단어가 나올 확률을 계산하게 됩니다.

$$ p(w_O|w_I) = \dfrac{\text{exp}(v'^\top_{w_O} v_{w_I})}{\sum^V_{w=1}\text{exp}(v'^\top_{w} v_{w_I})} $$

* $$ w_I $$ : 단어에 대한 one-hot vector
* $$ w_O $$ : surrounding words 로서 $$ w_{t+j} $$
* $$ V_{w} $$ : $$ W $$ 의 row vector를 가르킨다. 즉 input -> hidden matrix 를 가르킨다.
* $$ V'_{w} $$ : $$ W' $$ 의 column vector를 가르킨다. 즉 hidden -> output matrix 를 가르킨다. (dot product로 계산된 결과값)
* $$ V $$ : 단어의 갯수

$$ V'^\top_{w_O} $$ 를 학습할때 실제로는 "얼음" 이라는 단어의 one-hot vector가 들어갈것입니다.<br>
하지만 실제 추론시에는 hidden -> output matrix로 dot product가 계산된 결과값이며, 여기에는 여러 단어들이 나올 확률이 dense vector로 나올 것입니다. 그중에 "얼음" 이라는 해당 output vector안의 특정 element(실제로는 확률) 을 가르키는 것입니다.




## The Problem of Softmax

문제가 되는부분은 각각의 단어에 대해서 합을 구하는 denominator부분입니다. <br>
연산량 자체가 많기 때문에 실제로 최소 몇만단어에서 몇백만까지 vector의 크기가 늘어날수 있는 상황에서 softmax를 그대로 쓰는것은 문제가 있습니다.

또한 6만개의 단어와 300개의 neurons에 대해 위에서 언급한 skip-gram model을 사용하여 학습시.. <br>
input layer (6만 x 300) + output layer (300 x 6만) = 36000000 (bias 부분을 빼고도..) 뉴런들을 학습해야 합니다. <br>
실질적으로 위의 모델을 그대로 학습하려면 많은 양의 데이터와 많은 시간이 걸립니다.

Word2Vec의 저자는 이러한 문제를 인지하고 그의 [두번째](https://arxiv.org/pdf/1310.4546.pdf) 페이퍼에서 이 문제를 해결하는 방안에 대해서 3가지를 제안하고 있습니다.

1. 공통적인 단어나 문장을 하나의 단어처럼 사용
2. 빈번하게 나오는 단어들을 subsampling하여 training시에 학습률을 다른 단어와 맞춰줌으로서 class imbalance를 해결한다.
3. `Negative Sampling`이라는 테크닉을 통해서 각각의 training sample은 오직 모델의 업데이트를 제한하도록 한다.

결론적으로 softmax layer의 denominator부분의 연산량을 줄이는것과, sampling방법으로 최적화를 하는것이 word embedding models의 챌린지였습니다. 다음에 소개할 부분들을 이러한 문제들을 해결한 여러가지 방안에 대해서 소개를 합니다.







# Sampling-based Methods

Softmax-based 방법들은 기본적인 softmax의 구조를 유지하지만, sampling-based 방법들은 softmax layer자체를 사용하지 않습니다. <br>
즉 연산하기 쉬운 다른 loss들을 사용하여 softmax의 denominator의 normalization을 approximation합니다.<br>
하지만 Sampling-based 방법은 오직 training 중에만 효과가 있습니다. Inference 도중에는 여전히 full softmax 를 연산해서 normalised probability를 얻어야 합니다.

Sampling-based methods를 이해하기 위해서 softmax 에 cross-entropy를 사용했을때 loss function은 다음과 같이 됩니다.<br>
<small style="color:#777">
(cross-entropy loss는 $$ -\sum_{i=0} y^{(i)} \log \hat{y}^{(i)} $$ 인데 여기서 $$ y $$ 의 경우 one-hot vector로 들어가기 때문에 실질적으로 $$ -log \hat{y}^{(i)} $$ 가 되며 $$ \hat{y}^{(i)} $$ 는 softmax를 가르킵니다.)<br>
공식을 줄이기 위해서 $$ h^\top v^\prime_w $$ 를 [Bengio and Senecal](http://www.iro.umontreal.ca/~lisa/pointeurs/submit_aistats2003.pdf)에서 나온대로  $$ -\mathcal{E}(w) $$ 이렇게 줄여서 표현합니다.

논문에서 나온 내용..<br>
$$ \mathcal{E} \left( w_t, h_t \right) = b_{w_t} + V_{w_t} \cdot a $$

$$ V $$ : hidden to output layer weights

</small>

$$ J_\theta = -log \frac{exp(-\mathcal{E}(w))}{\sum_{w_i \in V} exp(-\mathcal{E}(w_i))} $$

$$ J_\theta $$ 는 전체 모든 단어에 대한 nagative log-probabilities 의 평균값입니다. <br>
미분을 좀더 빠르게 하기 위해서 $$ J_\theta $$를 decompose시킵니다.<br>
<small style="color:#777">
    참고로.. <br>
    $$ \log_a \left(\frac{x}{y} \right) = \log_a x - \log_a y $$   <br>
    $$ \log_a \left(xy \right) = \log_a x + \log_a b $$ <br>
</small>

$$ J_\theta = \mathcal{E}(w) + \log \sum_{w_i \in V} exp(-\mathcal{E}(w_i)) $$

Backpropagation을 위해서 gradient $$ \nabla $$ 를 구합니다.<br>
<small style="color:#777">
    $$ \nabla $$ 와 $$ \Delta $$ 의 차이는 다음과 같습니다.<br>
    $$ \Delta $$ : 두 points같의 차이를 가르키며 결과값은 scalar <br>
    $$ \nabla $$ : vectors상에서의 gradient값을 가르키며, 결과값은 vector<br>
    실제로는 거의 혼용해서 사용
</small>

$$ \nabla_\theta J_\theta = \nabla_\theta \mathcal{E}(w) + \nabla_\theta \log \sum_{w_i \in V} exp(-\mathcal{E}(w_i)) $$

Gradient값을 계산합니다. 이때 chain rule을 적용하게 됩니다.<br>
<small style="color:#777">
    Log Rules 은 다음과 같습니다.<br>
    $$ \frac{d}{dx} \ln(x) = \frac{1}{x}  $$ <br>
    $$ \frac{d}{dx} \log_a(x) = \frac{1}{x \ln(a)} $$
</small>

$$ \nabla_\theta J_\theta =  \nabla_\theta \mathcal{E}(w) + \frac{1}{\sum_{w_i \in V} exp(-\mathcal{E}(w_i))} \nabla_\theta \sum_{w_i \in V} exp(-\mathcal{E}(w_i))  $$

Gradient를 summation안쪽으로 옮겨줄수 있습니다.

$$ \nabla_\theta J_\theta =  \nabla_\theta \mathcal{E}(w) + \frac{1}{\sum_{w_i \in V} exp(-\mathcal{E}(w_i))}  \sum_{w_i \in V} \nabla_\theta exp(-\mathcal{E}(w_i))  $$

gradient of exp(x) 는 그대로 exp(x) 이며, 다시한번 chain rule을 타게 됩니다.

$$ \nabla_\theta J_\theta =  \nabla_\theta \mathcal{E}(w) + \frac{1}{\sum_{w_i \in V} exp(-\mathcal{E}(w_i))}  \sum_{w_i \in V}  exp(-\mathcal{E}(w_i)) \nabla_\theta \left( -\mathcal{E}(w_i) \right)  $$

위의 공식을 reposition하게 되면 다음과 같은 공식이 됩니다.

$$ \nabla_\theta J_\theta =  \nabla_\theta \mathcal{E}(w) + \sum_{w_i \in V} \frac{exp(-\mathcal{E}(w_i))}{\sum_{w_i \in V} exp(-\mathcal{E}(w_i))} \nabla_\theta \left( -\mathcal{E}(w_i) \right) $$

위의 공식을 보면 $$ \frac{exp(-\mathcal{E}(w_i))}{\sum_{w_i \in V} exp(-\mathcal{E}(w_i))} $$ 이 부분은 softmax probability 이며, $$ P(w_i) $$ (context $$ c $$ 는 간추리기 위해서 제거함) 로 바꿔써줄수 있습니다.

$$ \nabla_\theta J_\theta =  \nabla_\theta \mathcal{E}(w) - \sum_{w_i \in V} P(w_i) \nabla_\theta \mathcal{E}(w_i) $$

[Quick Training of Probabilistic Neural Nets by Importance Sampling](http://www.iro.umontreal.ca/~lisa/pointeurs/submit_aistats2003.pdf) 에서 Bengio and Senécal는 해당 gradient는 2개의 파트로 이루어 져있다고 언급합니다.  하나는 **positive reinforcement** 로서 target word $$ w $$ (위의 공식에서 첫번째 term), 그리고 두번째는 **negative reinforcement**로서 $$ V $$ 안의 모든 단어들에 대한 $$ \mathcal{E} $$ gradient의 expectation $$ \mathbb{E}_{w_i \sim P} $$ 을 가르킵니다.

$$ \sum_{w_i \in V} P(w_i) \nabla_\theta \mathcal{E}(w_i) =
\mathbb{E}_{w_i \sim P} \left[ \nabla_{\theta} \mathcal{E}(w_i)  \right]$$

Sampling-based 방법의 주요 목표는 바로 저 **negative reinforcement** 부분의 approximation을 찾아서 연산을 쉽게 하는 것입니다. <br>
즉 전체 단어에 대한 확률을 구하는 것을 피하고자 하는 것입니다.







# [Note] Information Theory

## Information Content

Information Content란 어떤 정보를 얻기위해 binary decisions의 횟수와 관련이 있습니다. <br>
N개의 elements를 갖고 있는 집합속에서 정확한 element를 찾아내는데 필요한 binary decisions의 횟수 <br>
(= 질문의 횟수이며 대답은 yes 또는 no) 는 다음과 같습니다.

$$ n_q = \log_2 N $$


예를 들어서 4개의 nucleotides {A, C, G, T} 가 있다면.. <br>
{% highlight text %}
    철수: Nucleotide를 머리속에 하나 갖고 있을때, 내가 뭘 생각하고 있을까?
    영희: A 또는 T이니?
    철수: 아니!
    영희: 그러면 G야?
    철수: 어!
{% endhighlight %}

따라서 $$ \log_2 4 = 2 $$ 개의 질문이 정확한 답을 찾기 위해서 필요로 합니다.<br>

만약 N개의 elements를 갖고 있다면, 찾고자 하는 element가 N/2안에 있는지 물어보고, 그다음 N/4안에 있는지 물어보고 계속이어나가서 답을 찾습니다. 즉 N=8이라면 각각의 element를 하나하나씩 찾는게 아니라, 반씩 줄여나가는 것입니다.

만약 각각의 element가 동일한 확률을 갖고 있다면 확률은 $$ p = \frac{1}{N} $$ 이 되며 공식으로는 다음과 같습니다.

$$ n_q = \log_2 N = - \log_2 p $$

예를 들어서 4개의 elements가 존재하고, 각각의 확률이 모두 동일하다면 다음과 같습니다.

$$ n_q = \log_2 8 = -\log_2 \frac{1}{8} = 3 $$


하지만 일반적으로 각각의 elements들은 모두 다른 확률을 갖고 있을 확률이 더 높습니다.<br>
1961년 Tribus는 "surprisal" $$ h_i $$ 이라는 컨셉으로 공식화합니다.

$$ h_i = -log_2 p_i $$

확률 $$ p_i = 1 $$ 이라면 $$ h_i = 0 $$ 일것이고, $$ p_i $$ 가 0에 가까워질수록 $$ h_i $$ 는 무한에 가까워질 것입니다.


<img src="{{ page.asset_path }}word2vec_surprisal.png" class="img-responsive img-rounded img-fluid">

## Entropy (Uncertainty Measure)

이러한 기초적 사실에 더하여, Shannon은 **Entropy 라 불리는 Uncertainty Measure**를 만들게 됩니다. <br>
아래 공식에서 $$ p_i $$ 는 얼마나 자주 발생하는지에 관한 확률이라고 볼수 있습니다.

$$ H = \sum_i p_i h_i = -\sum p_i log_2 p_i $$

아래는 Scipy에서 제공하는 entropy와 제가 만든 entropy의 구현입니다. <br>
Scipy가 사용하는 공식은 $$ S = -\sum p_i * \log p_i $$ 로서 natural log를 사용합니다.
따라서 결과값이 위에서 말한 공식과는 좀 다릅니다.


{% highlight python %}
def my_entropy(x):
    x = np.array(x)
    return -np.sum(x * np.log2(x))

print('entropy   :', entropy([0.25, 0.25, 0.25, 0.25]))
print('my entropy:', my_entropy([0.25, 0.25, 0.25, 0.25]))

# 불균형한 분포의 경우 예측가능성이 더 높아지기 땜누에 엔트로피도 더 줄어든다
print()
print('불균형한 분포 entropy   :', entropy([0.1, 0.1, 0.1, 0.7]))
print('불균형한 분포 my entropy:', my_entropy([0.1, 0.1, 0.1, 0.7]))
{% endhighlight %}

{% highlight text %}
entropy   : 1.38629436112
my entropy: 2.0

불균형한 분포 entropy   : 0.940447988655
불균형한 분포 my entropy: 1.35677964945
{% endhighlight %}