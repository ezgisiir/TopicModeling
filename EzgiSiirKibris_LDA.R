#############################################
####  Topic Modelling                  #####
####  US presidential Inaugural Speeches###
####   Ezgi Siir Kibris                ####
#### 13 February 2024                 ####
##########################################



# Sources:

# https://sicss.io/2019/materials/day3-text-analysis/topic-modeling/rmarkdown/Topic_Modeling.html#limitations-of-topic-models

# https://content-analysis-with-r.com/6-topic_models.html

# https://www.youtube.com/watch?v=4YyoMGv1nkc&ab_channel=KasperWelbers

# UN General Debates Dataset: https://www.kaggle.com/datasets/unitednations/un-general-debates



### LDA Topic Model ###

# Presidential inaugural speeches #

# Data Preprocessing

library(quanteda) 
corp = corpus_reshape(data_corpus_inaugural, to = "paragraphs")
dfm = dfm(corp, remove_punct=T, remove=stopwords("english"))
dfm = dfm_trim(dfm, min_docfreq = 5)

library(topicmodels)
dtm = convert(dfm, to = "topicmodels") 
set.seed(1)
m = LDA(dtm, method = "Gibbs", k = 10,  control = list(alpha = 0.1))



terms(m, 5)

topic = 6
words = posterior(m)$terms[topic, ]
topwords = head(sort(words, decreasing = T), n=50)
head(topwords)


# Wordclouds 

library(wordcloud)
wordcloud(names(topwords), topwords)


topic.docs = posterior(m)$topics[, topic] 
topic.docs = sort(topic.docs, decreasing=T)
head(topic.docs)


topdoc = names(topic.docs)[1]
topdoc_corp = corp[docnames(corp) == topdoc]
texts(topdoc_corp)


# Top words

library(tidytext)
library(dplyr)
library(ggplot2)

president_topics <- tidy(m, matrix = "beta")

president_top_terms <- 
  president_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

president_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()




# Evaluation

# LDA vis produces html
# Circles are topic. when we click we get most common words and their distribution.
# when circles are closer maybe we need less topics
# when they are further apart maybe we need more topics
# Hover around topics and words

library(LDAvis)   

dtm = dtm[slam::row_sums(dtm) > 0, ]
phi = as.matrix(posterior(m)$terms)
theta <- as.matrix(posterior(m)$topics)
vocab <- colnames(phi)
doc.length = slam::row_sums(dtm)
term.freq = slam::col_sums(dtm)[match(vocab, colnames(dtm))]

json = createJSON(phi = phi, theta = theta, vocab = vocab,
                  doc.length = doc.length, term.frequency = term.freq)
serVis(json)


############################################################################################

### STM Topic Model ###

## UN General Debates ##

# Sources:

# https://sicss.io/2019/materials/day3-text-analysis/topic-modeling/rmarkdown/Topic_Modeling.html#limitations-of-topic-models

# https://content-analysis-with-r.com/6-topic_models.html

# Dataset: https://www.kaggle.com/datasets/unitednations/un-general-debates


library(tidyverse)
library(ldatuning)
library(stm)
library(tidytext)

theme_set(theme_bw())

setwd("C:/Users/ezgim/Desktop/archive")

df=read.csv("un-general-debates.csv")
df2014=subset(df, subset=df$year>'2014') 

# Data preprocessing

#Remove punctuation, stopwords ,numbers,...

dfm.un <- dfm(df2014$text, remove_numbers = TRUE, remove_punct = TRUE, remove_symbols = TRUE, remove = stopwords("english"))

# Term frequency, keep words according to their frequency in the data, 
# set maximum and minimun frequency

dfm.un.trim <- dfm_trim(dfm.un, min_docfreq = 0.075, max_docfreq = 0.90, docfreq_type = "prop")

library(stm)
?stm

### UN STM

n.topics <- 40
dfm2stm <- convert(dfm.un.trim, to = "stm")

modell.stm <- stm(dfm2stm$documents, dfm2stm$vocab, K = n.topics, data = dfm2stm$meta, init.type = "Spectral")


as.data.frame(t(labelTopics(modell.stm, n = 10)$prob))




par(mar=c(0.5, 0.5, 0.5, 0.5))
cloud(modell.stm, topic = 1, scale = c(2.25,.5))


cloud(modell.stm, topic = 3, scale = c(2.25,.5))


cloud(modell.stm, topic = 7, scale = c(2.25,.5))

cloud(modell.stm, topic = 9, scale = c(2.25,.5))


plot(modell.stm, type = "summary", text.cex = 0.5, main = "Topic shares on the corpus as a whole", xlab = "estimated share of topics")



plot(modell.stm, type = "hist", topics = sample(1:n.topics, size = 9), main = "histogram of the topic shares within the documents")


plot(modell.stm, type = "labels", topics = c(5, 12, 16, 21), main = "Topic terms")


plot(modell.stm, type = "perspectives", topics = c(16,21), main = "Topic contrasts")

#####################################################################################


