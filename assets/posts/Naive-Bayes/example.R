#install.packages('tm') # Text Mining Library
#install.packages('wordcloud')
#install.packages('e1071')
#install.packages('gmodels')
library(wordcloud)
library(tm)
library(e1071)
library(gmodels)

# sms <- read.delim('data.csv', header=F, sep='\t', )
colnames(sms) <- c('type', 'text')
sms$text <- as.vector(sms$text)

corpus <- Corpus(VectorSource(sms$text))
corpus.clean <- tm_map(corpus, content_transformer(tolower))
corpus.clean <- tm_map(corpus.clean, removeNumbers)
corpus.clean <- tm_map(corpus.clean, removeWords, stopwords())
corpus.clean <- tm_map(corpus.clean, removePunctuation)
corpus.clean <- tm_map(corpus.clean, stripWhitespace)
dtm <- DocumentTermMatrix(corpus.clean)

sms_raw.train <- sms[1:as.integer(nrow(sms) * 0.75), ]
sms_raw.test <- sms[as.integer(nrow(sms) * 0.75):nrow(sms), ]

dtm.train <- dtm[1:as.integer(nrow(dtm) * 0.75),]
dtm.test <- dtm[as.integer(nrow(dtm) * 0.75):nrow(dtm), ]

corpus.train <- corpus.clean[1:as.integer(length(corpus.clean) * 0.75)]
corpus.test <-  corpus.clean[as.integer(length(corpus.clean) * 0.75): length(corpus.clean)]

wordcloud(subset(sms, type=='spam')$text, min.freq = 40, random.order = F)
wordcloud(subset(sms, type=='ham')$text, min.freq = 40, random.order = F)

sms.train <- DocumentTermMatrix(corpus.train, list(findFreqTerms(dtm.train, 5)))
sms.test <-  DocumentTermMatrix(corpus.test, list(findFreqTerms(dtm.test, 5)))


convert_counts <- function(x){
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels=c(0, 1), labels=c('No', 'Yes'))
  return (x)
}

sms.train <- apply(sms.train, MARGIN=2, convert_counts)
sms.test <- apply(sms.test, MARGIN=2, convert_counts)


sms_classifier <- naiveBayes(sms.train, sms_raw.train$type)
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
