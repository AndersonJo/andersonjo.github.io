#install.packages('tm') # Text Mining Library
library(tm)

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

sms.train <- sms[1:as.integer(nrow(sms) * 0.75), ]
sms.test <- sms[as.integer(nrow(sms) * 0.75):nrow(sms), ]

dtm.train <- dtm[1:as.integer(nrow(dtm) * 0.75),]
dtm.test <- dtm[as.integer(nrow(dtm) * 0.75):nrow(dtm), ]

corpus.train <- corpus.clean[1:as.integer(nrow(corpus.clean) * 0.75), ]
