---
layout: post
title:  "Naive Bayes"
date:   2015-08-04 02:00:00
categories: "machine-learning"
asset_path: /assets/posts/Naive-Bayes/

---
Bayes 공식이나 이론은 이미 [Monty Hall][bayes] 문제를 풀면서 설명을 했습니다.

오늘은 이 Bayes 공식을 이용한.. 정말 간단하지만 왠만한 복잡도 높은 다른 classification methods 보다 강력한 퍼포먼스(정확도)를
보여주는 Naive Bayes를 예제와 함께 설명하겠습니다.

일단 Naive Bayes의 공식은 다음과 같습니다.

<img src="{{page.asset_path}}naive-bayes-formula.gif" class="img-responsive img-rounded">

1. C<sub>L</sub> 은 클래스 또는 분류를 나타냅니다.
2. F 는 features
3. 1/Z 는 scaling factor

위의 공식 맨 오른쪽에 보면 P(F|C) 요런게 있는데 결국 conditional probability입니다.<br>
하지만 Naive Bayes는 각각의 multiple 기호 안에 있는 likelihood 들이 모두 independent하다고 가정을 합니다.
즉 서로 영향을 미치지 않고 동전 앞이 나올지 뒤가 나올지 여러번 던지듯이 서로 연관성이 없다고 가정을 하고 들어갑니다.

이유는 서로간에 dependent하다면 계산방식이 매우 복잡해지며 효율적인 알고리즘으로 사용하기 적합하지 않기 때문입니다.
그럼에도 Naive Bayes는 왠만한 복잡도 있는 machine learning 알고리듬에 비해서 매우 강력한 성능을 보여줍니다.



<img src="{{page.asset_path}}what-the-fuck.jpg" class="img-responsive img-rounded">

대체 뭔 소리여 ㅋㅋㅋ<br> 
그냥 예제하나 풀면 다 이해됩니다.


# References

1. [데이터 다운로드][data]<br>
   *데이터는 [Unicamp][unicamp] 에서 가져왔습니다.*
2. [Machine Learning With R][book]

# SMS Spam Classification

문자중에는 스팸 문자도 있고, 친구한테서 온 문자들도 있습니다.<br>
아마도 스팸문자 안에는 '할인', '비아그라', '세일' 같은 단어들이 많을거라 생각됩니다.<br>
친구나 연인끼리의 문자를 보냈다면 '지금', '집', '어디', '간다', '사랑' 같은 단어들이 좀 더 많이 있지 않을까 추측해 봅니다.

자! 포인트는 이러한 스팸에서 나오는 단어들 그리고 친구나 연인끼리의 문자에서 나오는 단어들을 features로 삼아서 
확률적으로 해당 문자가 스팸인지 또는 햄(ham : 스팸이 아닌 문자)인지 알아낼수 있지 않을까요?

중요 포인트는 '할인', '비아그라', '세일' 같은 단어가 나왔다고 무조건 스팸문자는 아니고, <br>
또한 '집', '어디', '간다', '사랑' 이라는 단어가 나왔다고 무조건 친구나 연인이 보낸 문자로 가정 지으면 안됩니다.

아래 예제에서는 스팸에서 나온 문자들은 스팸일 확률을 높이는 단어들로 묶고, 햄에서 나온 단어들은 스팸이 아닌 확률로 좀더 높입니다.




일단 예제에서 사용되는 용어부터 정의하겠습니다.

| 용어 | 내용 |
|:--|:--|
| spam | 스팸문자 |
| ham | 스팸이 아닌 문자


라이브러리들을 설치및 import해 줍니다.
{% highlight r%}
install.packages('tm') # Text Mining Library
install.packages('wordcloud')
install.packages('e1071')
install.packages('gmodels')
library(wordcloud)
library(tm)
library(e1071)
library(gmodels)
{% endhighlight %}

데이터를 읽습니다.

{% highlight r%}
sms <- read.delim('data.csv', header=F, sep='\t', )
colnames(sms) <- c('type', 'text')
sms$text <- as.vector(sms$text)
{% endhighlight %}


그 다음으로 Data Preparation이 필요합니다. <br>
문자안에 들어있는 단어들을 통해서 Bayes 확률적으로 이것이 spam 인지 ham인지 알기 위해서는 필요없는 단어들은 제거해주는게 좋습니다.
예를 들어서 but, at, were, was, in 같은 단어들은 그닥 필요가 없으며, 숫자같은 경우 숫자만으로는 spam인지 ham인지 구분짓기가 힘들기 때문에..
숫자또한 제거해 줍니다. 

참고로 corpus는 하나의 글 뭉탱이라고 생각하면 되며, 그냥 sms$text 전체 말뭉치라고 생각하면 됩니다.

{% highlight r%}
corpus <- Corpus(VectorSource(sms$text))
corpus.clean <- tm_map(corpus, content_transformer(tolower))
corpus.clean <- tm_map(corpus.clean, removeNumbers)
corpus.clean <- tm_map(corpus.clean, removeWords, stopwords())
corpus.clean <- tm_map(corpus.clean, removePunctuation)
corpus.clean <- tm_map(corpus.clean, stripWhitespace)
dtm <- DocumentTermMatrix(corpus.clean)
{% endhighlight %}

그 다음으로 75% 데이터는 트레이닝용으로 그리고 나머지는 Naive Bayes를 이용한 스팸 분류가 잘 되는지 퍼포먼스 테스트용으로 사용하겠습니다.

{% highlight r%}
sms_raw.train <- sms[1:as.integer(nrow(sms) * 0.75), ]
sms_raw.test <- sms[as.integer(nrow(sms) * 0.75):nrow(sms), ]

dtm.train <- dtm[1:as.integer(nrow(dtm) * 0.75),]
dtm.test <- dtm[as.integer(nrow(dtm) * 0.75):nrow(dtm), ]

corpus.train <- corpus.clean[1:as.integer(length(corpus.clean) * 0.75)]
corpus.test <-  corpus.clean[as.integer(length(corpus.clean) * 0.75): length(corpus.clean)]
{% endhighlight %}

단어들을 구름 뭉탱으로 봅시다.

{% highlight r%}
wordcloud(subset(sms, type=='spam')$text, min.freq = 40, random.order = F)
wordcloud(subset(sms, type=='ham')$text, min.freq = 40, random.order = F)
{% endhighlight %}


<div style="text-align:center; font-size:2em;">Spam</div>
<img src="{{page.asset_path}}spam.png" class="img-responsive img-rounded">

<div style="text-align:center; font-size:2em;">Ham</div>
<img src="{{page.asset_path}}ham.png" class="img-responsive img-rounded">

DocumentTermMatrix는 각각의 문자들 마다 전체 단어중 몇개가 들어 있는지를 나타냅니다.<br>
예를 들어서. ... 다음과 같이 나올수 있습니다. 

| sales | free | now | love |
|:-----|:-----|:-----|:-----|
| 2 | 1 | 0 | 0|
| 0 | 0 | 1 | 2|
| 0 | 1 | 0 | 0|

inspect(sms.train) 해보면 정확하게 볼 수 있는데.. 각각의 SMS 문자마다 전체 단어 중에서 몇개 나왔는지 보여주기 때문에.. 
R 에서 보면 좀 난감한게 좀 있습니다.

findFreqTerms 같은경우는 전체중에서 5번 이상 나온 단어들만 사용해서 분석하겠다는 뜻입니다.

{% highlight r%}
sms.train <- DocumentTermMatrix(corpus.train, list(findFreqTerms(dtm.train, 5)))
sms.test <-  DocumentTermMatrix(corpus.test, list(findFreqTerms(dtm.test, 5)))
{% endhighlight %}


apply함수를 통해서 각각의 메세지를 'Yes', 'No' 만 갖고있는 factor로 만들어줌

\* 여리서 MARGIN=1 은 columns를 가르키고, MARGIN=2는 rows를 가르킵니다. 


{% highlight r%}
convert_counts <- function(x){
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels=c(0, 1), labels=c('No', 'Yes'))
  return (x)
}

sms.train <- apply(sms.train, MARGIN=2, convert_counts)
sms.test <- apply(sms.test, MARGIN=2, convert_counts)
{% endhighlight %}

# Training and Predicting a model

e1071 라이브러리를 사용해서 naive bayes를 처리하도록 하겠습니다.



{% highlight r%}
sms_classifier <- naiveBayes(sms.train, sms_raw.train$type, laplace=1)
sms_predictions <- predict(sms_classifier, sms.test)

CrossTable(sms_predictions, sms_raw.test$type)

   Cell Contents
|-------------------------|
|                       N |
| Chi-square contribution |
|           N / Row Total |
|           N / Col Total |
|         N / Table Total |
|-------------------------|

 
Total Observations in Table:  1394 

 
                | sms_raw.test$type 
sms_predictions |       ham |      spam | Row Total | 
----------------|-----------|-----------|-----------|
            ham |      1200 |        16 |      1216 | 
                |    19.277 |   128.373 |           | 
                |     0.987 |     0.013 |     0.872 | 
                |     0.990 |     0.088 |           | 
                |     0.861 |     0.011 |           | 
----------------|-----------|-----------|-----------|
           spam |        12 |       166 |       178 | 
                |   131.691 |   876.974 |           | 
                |     0.067 |     0.933 |     0.128 | 
                |     0.010 |     0.912 |           | 
                |     0.009 |     0.119 |           | 
----------------|-----------|-----------|-----------|
   Column Total |      1212 |       182 |      1394 | 
                |     0.869 |     0.131 |           | 
----------------|-----------|-----------|-----------|
{% endhighlight %}


[bayes]: /machine-learning/2015/07/29/Monty-Hall-Problem/
[data]: {{page.asset_path}}data.csv
[book]: http://www.amazon.com/Machine-Learning-R-Brett-Lantz/dp/1782162143/ref=sr_1_1?ie=UTF8&qid=1438737465&sr=8-1&keywords=machine+learning+with+r
[unicamp]: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

