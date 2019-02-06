---
layout: resume
title:  "Twitter Sentiment Analysis"
date:   2018-12-24 00:00:00
categories: jekyll
---

# Sentiment analysis of automotive Tweets

The automotive industry is a multi-buillion dollar company relying on postive business-client relations. With so much money at stake, 
it is critical for these types of businesses to monitor and protect their reputation. One way of obtaining live data about is to use Twitter
to monitor the information that customers are tweeting about businesses.

In this blog post, we outline the process building the sentiment classification model, data collection processing and storage,
topic modelling and finally we give the results of our analysis.

# Contents

1. [**Sentiment analysis model**](#1-sentiment-analysis-model)  
a. BOW (Bag of Words)   
b. Sentence Preprocessing  
c. Model training and Selection

2. [**Data collection (Twitter)**](#2-data-collection)  
a. SQL table structure  
b. Twitter Listener Code  
c. Topic Modelling (examples)   

3. [**Results**](#3-results)  
a. Overall Sentiments for Top Car brands  
b. Sentiment Time series vs Stock Price  
c. Linear Model Fit



# 1. Sentiment analysis model
Next we trained several binary classification models and settled 


The data sets that we used were:
1. reviews_Automotive_5.json
2. reviews_Office_Products_5.json
3. yelp.csv
4. training.1600000.processed.noemoticon.csv

From these various data sets, we read in and processed the **review** and **rating** categories. 

```
---------------------------------------------
Total Yelp reviews:  10000
Total Amazon automotive reviews:  20473
Total Amazon office reviews:  53258
Total Twitter Data reviews:  70000
---------------------------------------------
```

The ratings of the Amazon and yelp reviews were on a scale of 1-5 and there were different amounts
in each category.

```
Raw reviews: 
1-Star reviews: 2421
2-Star reviews: 3259
3-Star reviews: 7951
4-Star reviews: 57340
5-Star reviews: 47592
```

After balacing the data set, we have the following distribution of reviews,
Therefore, using pandas, we were able to balace the reviews giving us the following distribution

```
Balaced reviews: 
1-Star reviews: 2421
2-Star reviews: 2421
3-Star reviews: 2421
4-Star reviews: 2421
5-Star reviews: 2421
```
After balacing the data set




```
================================================================

Train data set size:  2800 

Test data set size:  560 

================================================================

Baseline Model: 
              precision    recall  f1-score   support

         -1       0.53      0.55      0.54       283
          1       0.52      0.49      0.50       277

avg / total       0.52      0.52      0.52       560

================================================================

Naive Bayes: 
              precision    recall  f1-score   support

         -1       0.87      0.77      0.82       283
          1       0.79      0.88      0.83       277

avg / total       0.83      0.83      0.83       560

================================================================

Desicion Tree: 
              precision    recall  f1-score   support

         -1       0.74      0.71      0.72       283
          1       0.71      0.74      0.73       277

avg / total       0.73      0.72      0.72       560

================================================================

Random Forests: 
              precision    recall  f1-score   support

         -1       0.81      0.81      0.81       283
          1       0.80      0.80      0.80       277

avg / total       0.81      0.81      0.81       560

================================================================

Logistic Regression: 
              precision    recall  f1-score   support

         -1       0.88      0.84      0.86       283
          1       0.85      0.89      0.87       277

avg / total       0.87      0.86      0.86       560

================================================================
```

Of these models, the Logistic regression and Naive Bayes does the best, so we choose these as our two Sentiment models.


# 2. Data collection
In order to collect twitter an AWS EC2 instance was set up to collect the data. 


## c. Topic modelling

The twitter data contains alot of different topics that 


# 3. Results
## a. overall sentiments for the top car brands 
