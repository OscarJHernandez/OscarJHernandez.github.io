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
================================================================
Raw reviews: 
1-Star reviews: 2421
2-Star reviews: 3259
3-Star reviews: 7951
4-Star reviews: 57340
5-Star reviews: 47592
================================================================
```

After balacing the data set, we have the following distribution of reviews,
Therefore, using pandas, we were able to balace the reviews giving us the following distribution

```
================================================================
Balaced reviews: 
1-Star reviews: 2421
2-Star reviews: 2421
3-Star reviews: 2421
4-Star reviews: 2421
5-Star reviews: 2421
================================================================
```
After balacing the data set the tweets are cleaned up using the 
**clean_up_text(text)** function. This function would remove punctuations,
numbers, captitalization.

After this process is complete,




The final results for the random baseline model, along with the 
Naive Bayes and Logistic regression model are shown below,

```
================================================================

Train data set size:  7747 

Test data set size:  1937 

================================================================

Baseline Model: 
              precision    recall  f1-score   support

         -1       0.53      0.55      0.54       283
          1       0.52      0.49      0.50       277

avg / total       0.52      0.52      0.52       560

================================================================

Naive Bayes: 
              precision    recall  f1-score   support

         -1       0.80      0.84      0.82       954
          1       0.84      0.79      0.81       983

avg / total       0.82      0.82      0.82      1937

================================================================

Logistic Regression: 
              precision    recall  f1-score   support

         -1       0.82      0.88      0.85       954
          1       0.87      0.82      0.84       983

avg / total       0.85      0.85      0.85      1937

================================================================
```
For our model, we used 


# 2. Data collection
In order to collect twitter an AWS EC2 instance was set up to collect the data. 


## c. Topic modelling

The twitter data contains alot of different topics that 


# 3. Results
## a. overall sentiments for the top car brands 
