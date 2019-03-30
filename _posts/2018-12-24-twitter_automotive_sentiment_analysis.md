---
layout: blog
title:  "Twitter Sentiment Analysis"
date:   2018-12-24 00:00:00
categories: jekyll
---
<style type="text/css" rel="stylesheet">
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 60%;
}

/* Three image containers (use 25% for four, and 50% for two, etc) */
.column {
  float: left;
  width: 40.0%;
  padding: 5px;
}

/* Clear floats after image containers */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>

# Sentiment analysis of automotive Tweets

The automotive industry is a multi-buillion dollar company relying on postive business-client relations. With so much money at stake, 
it is critical for these types of businesses to monitor and protect their reputation. One way of obtaining live data about is to use Twitter
to monitor the information that customers are tweeting about businesses.

In this blog post, we outline the process building the sentiment classification model, data collection processing and storage,
topic modelling and finally we give the results of our analysis.

# Contents

1. [**Sentiment analysis model**](#sentiment-analysis-model)  

2. [**Data collection and cleaning**](#data-collection-and-cleaning)    

3. [**Results**](#results)  
a. [Combining the sentiment curves](#combining-the-sentiment-curves)   
b. [Analysing stock returns and sentiment signals](#analysing-stock-returns-and-sentiment-signals)      
c. [Posteriors of the correlations](#posteriors-of-the-correlations)

4. [**Conclusions and Further Work**](#conclusions-and-further-work)


# Sentiment Analysis Model
The natural language model that we used in this project was the __Bag-of-words model (BOW)__. 
Using a large corpus of documents, the frequency of all occurring words is analysed and ranked according to the frequently occuring words. 
Then a function is used that maps a given sentence to a vector representing the occurrence of words from the vocabulary in the sentence.  
 
More sophisticated language models exist on the market, however as shown in [Ref. 1](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf), this simple BOW model is already capable of achieving about an 80%
 accuracy in sentiment identification problems, making it a good initial model for this pilot study.

Based on this __BOW__ model, we generated feature vectors of all corpus in the training set that consisted of __amazon reviews__, __yelp reviews__, 
and the __sentiment140__ data set. These feature vectors and their corresponding labels were used to train two machine learning binary classifier models,
 the __logistic regression (LR)__ and naive __Bayes classifiers (NB)__.
 
While more sophisticated sentiment classifier models exist, we chose to use these simpler approaches because they both provide
 probabilities in their categorizations that we will use as uncertainty estimates and it was indicated by [Max Margenot](https://www.quantopian.com/posts/learn-how-to-build-a-model-in-python-to-analyze-sentiment-from-twitter-data) that a sentiment
  classifier model based on __logistic regression__ method achieved about a 78% accuracy, while a more sophisticated __LSTM neural network__ scored 
  about 81.5% on the sentiment140 dataset and took significantly longer to train.
 This indicated to us that the __NB__ and __LR__ were sufficient.


<figure>
    <img src="/assets/img/model_metrics.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: Training metrics.
    </figcaption>
</figure>


Once these two models were trained, the final model that we used consisted of an ensemble of the LR, NB and pre-built textblob classifiers. 
The final output of the model was the sentiment that corresponded to the classifier that had the greatest confidence in the result that was also larger than the threshold value T that was used as input. This threshold value would be varied from 0 (indicating that the models could give predictions with 0% confidence in their values),
 up to the value 1 (where the model was only allowed to give a prediction if it was 100% confident in its predictions). 
 
 <figure>
    <img src="/assets/img/true_positives_ensemble_model.png" width="40%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: Percentage of identified true positive and false positive results for the ensemble model for
    different thresholds.
    </figcaption>
</figure>



# Data collection and cleaning  

With the model now in place, the next step of the project was to collect data from twitter based on keywords that are related 
to the car brands that we were examining. For this task we used the [twython library](https://twython.readthedocs.io/en/latest/) that
 allowed us the ability to listen to the live twitter feed based on the keywords that we used. 
 
An [AWS EC2](https://aws.amazon.com/ec2/) server was used to collect all of these tweets into an [SQLite database](https://www.sqlite.org/index.html). Once the twitter data was collected the database was 
downloaded and the text was processed as follows:
* punctuation was removed,
* text was coverted to lower case,
* URLS were removed
* "_" was removed,
* "RT" was removed,
* "@" sign was removed, 
* hashtags were removed, 
* numbers were removed, 
* text was lemmatized. 

We also removed duplicates from the data set. For any days where we had missing data, such as weekends for the stock market returns, 
the median of the data was used to fill the missing points. The histogram of the tweet length before the preprocessing steps is given in the figure below

<figure>
    <img src="/assets/img/raw_tweet_length.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: Histogram of tweet length in the data set before data cleaning.
    </figcaption>
</figure>

<br>
After processing the tweets and removing duplicates the histogram looks more Gaussian,
<br>

<figure>
    <img src="/assets/img/filtered_tweet_length.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: Histogram of tweet length after data cleaning.
    </figcaption>
</figure>


Another issue in the data set was that for specific car brands, 
their tweets contained many things not related to cars at all. To try to filter out irrelevant tweets, we used [topic modelling](https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24)
 on the database, with an optimal value for the __“number of topics”__ parameter in the prebuilt (latent dirichlet allocation or non-negative matrix factorization (NNM)) 
functions that we tried. This optimal number of topics model was determined by finding the number of topics $$(k)$$ that maximized the average cohesion score for all 
sub-topics (0 to 1). The cohesion function acts on a string by computing the average cosine similarity score ([using a prebuilt word embedding model from spacy **'en_core_web_lg'**](https://spacy.io/)) 
for all pairs of words in that string. 

{% highlight python linenos %}
def string_cohesion(vec):
	'''
	This function will compute the cohesion of a vector of words. This function
	is useful for choosing the number of topics for topic modelling
	
	
	The function has the following bounds:
	
	0<= string_cohesion(x) <=1
	
	string_cohesion(x) = 1
	when x consists of the same words
	
	string_cohesion(x) = 0
	when x is an empty string or there are no tokens for it
	'''
	s=0.0
	
	tokens = nlp(vec)
	Norm = len(tokens)
	
	# If we have a non-zero norm
	if(Norm!=0):
		
		for word1 in tokens:
			for word2 in tokens:
				if(word1.has_vector==True and word2.has_vector == True):
					s+= word1.similarity(word2)	
		s/=Norm**2
	
	return s
{% endhighlight %}
 
We used this method to find specific topics in the twitter data base that did not have anything to do with cars, for example some car tweets were related to people warning each other about police chasing a  criminal in a specific vehicle.

# Results

Finally after building the model, creating the AWS server to collect the tweets, and cleaning the data 
can conduct the analysis. Using SQL queries, we processed the tweets of specific car brands day-by-day 
for the two-week period using different model thresholds. These day-by-day sentiment 
of the tweets is shown below for Tesla,

<figure>
    <img src="/assets/img/Tesla_model_pos_threshold_variation.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: The day-by-day positive sentiment time series for Tesla for various thresholds, in terms of the 
    total percentage of the daily tweets.
    </figcaption>
</figure>

In the Figure above we see that appears to be a spike around August 8th for tesla, but as the models prediction certainty 
threshold value ($$T$$) increases, which the strengths of these spikes shrinks. We observe that the percentages of positive or negative 
tweets drops as we increase the certainty threshold of the model. 

## Combining the sentiment curves

In Fig XXX we observe that there are specific peaks in these sentiment curves that do not dissapear when you increase the certainty threshold. 
These are the signals that we want to extract from the data. However, we notice that the curves for different $$T$$ values have vastly different scales, 
the scale for the $$T=0$$ curves are around 30%, while for $$T$$=0.8 it's more like 3%. 

Another problem that we need to consider is that not all of the curves are to be taken with the same level of confidence. 

So now we are faced with two problems that we need to solve:

* __Is it possible to remove the scale associated with the different thresholds?__
* __Once we have similarly scales data, how can we combine the different curves together to get unique signal?__   

This first problem can be solved easily by __normalizing__ the data.
We chose to do this normalization with the __Z-transform__ for each curve obtained using the different thresholds, this will
give us the set of curves  

$$
\\ \lbrace \text{Pos}(x_t,T), \text{Neg}(x_t,T) \rbrace \\
$$  

for the positive and negative sentiment signals, where $$x_t$$ represents the day, and $$T$$ the threshold.

Now we need to know how to combine them. The solution to this inspired by Bayesian statistics. The positive and negative sentiment 
curves for different thresholds are combined into a final true positive and false positive curves, denoted $$P(x_t)$$
and $$\tilde{P}(x_t)$$, respectively, are

$$
\\  P(x_t) = \int\limits_0^1 dT \ \text{Pos}(x_t,T)W_{TP}(T),  \\  
$$

$$
\\ \tilde{P}(x_t) = \int\limits_0^1 dT \ \text{Pos}(x_t,T)W_{FP}(T). \\
$$  

where $$W_{TP}(T)$$, $$W_{FP}(T)$$ are the true-positive and false-positive weights at a given $$T$$. These weights 
represent how seriously we should take a given sentiment curve into account for a specific $$T$$. Their exact expressions are

$$
\\ W_{TP}(T) = P(\hat{y}=1|y=1,T)P(y=1|T)P(T), \\
$$

$$
\\ W_{FP}(T) = P(\hat{y}=1|y=-1,T)P(y=-1|T)P(T). \\
$$

Here, the symbol $$\hat{y}$$ represents the estimate of the sentiment from the model, $$y$$ is the true sentiment value. Therefore, 
$$P(\hat{y}=1|y=1,T)$$ indicates the true positive probability, $$P(y=1|T)$$ is the prior probability of positive tweets, and $$P(T)$$
is the prior distribution for the threshold. 

Similar expressions hold for the true negative, and false negative sentiment signals. For the incoming 
automotive tweets, we do not know the true accuracies for positive and negative sentiments but we will make the assumption that the scores 
obtained from the testing set will be the same as for the live automotive tweets. If the model has been well trained, 
then we would expect our model to generalize and this assumption should be approximately correct. In the figure below



<figure>
    <img src="/assets/img/Pos_signal_weight.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: The negative sentiment time series for Tesla. The dark/light red band 
    indicate the estimated True negative/ False Negative signals, respectively.
    </figcaption>
</figure>

<br>

<figure>
    <img src="/assets/img/Neg_signal_weight.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: The positive sentiment time series for Tesla. The dark/light green band 
    indicate the estimated True negative/ False Negative signals, respectively.
    </figcaption>
</figure>

<br>
<br>
Using the Bayesian weighting average formula that we found earlier, we obtain our normalized positive and 
negative sentiment signals.
<br>
<br>

<figure>
    <img src="/assets/img/Normed_pos_signal.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: The positive sentiment time series for Tesla. The dark/light green band 
    indicate the estimated True negative/ False Negative signals, respectively.
    </figcaption>
</figure>

<figure>
    <img src="/assets/img/Normed_neg_signal.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: The negative sentiment time series for Tesla. The dark/light red band 
    indicate the estimated True negative/ False Negative signals, respectively.
    </figcaption>
</figure>

<br>

In the next section, we will analyse this data within a specific financial framework to look for correlations between these sentiment signals
and the return of the stocks for specific vehicle brands.


## Analysing stock returns and sentiment signals  

Now that we normalized the data and combined the curves at different thresholds together into a single sentiment signal time series, we want to study the correlation between this time-series and the stock return time series of the stock. But the next problem that we encounter is to understand: how they should be related?

Thankfully was not overly complicated, [there is a financial model, known as the CAPM](https://www.investopedia.com/terms/c/capm.asp)(note we use 0 for the risk free return rate) that relates the return of an asset to the return of the market,
 through a linear model. The base CAP model states that the return of a stock $r$ should be proportional to the return of the market $$r_{M}$$   
 
$$
\\ r(x_t) = \alpha + \beta_{M} r_{M}(x_t), \\
$$

Here the constant of proportionality to the market is $$\beta_M$$, and $$\alpha$$ is a small offset, which should be zero. This base hypothesis can 
be easily extended to include additional coefficients $${\bf \beta_N}$$ and $$\beta_M$$ factors
that allow us to incorperate the effects of the positive and negative sentiment signals that we found earlier 

$$
\\ r(x_t) = \alpha + \beta_{M} r_{M}(x_t) + \beta_{N} N(x_t)  + \beta_{P} P(x_t) \\
$$

In this case, we need to include a factor $$\beta_N$$ and $$\beta_P$$ that will account for the influence of the true positive and true negative 
time series signals. 

With the computed and normalized positive and negative sentiment signals, we now carry out a Bayesian linear regression analysis to determine
the distributions of the linear fit coefficients. In the Bayesian framework, we need to determine the posterior distribution of a parameter
$$\beta$$ given the times series data, denoted $$D$$. This posterior distribution is denoted $$P(\beta|D)$$. 
 
This Bayesian model fitting was carried out using the __[emcee Python package](http://dfm.io/emcee/current/)__, 
that employs a parallelized Markov-Chain monte carlo algorithm to compute the posterior distributions of the parameters of interest.
Mathematically speaking we are carrying out the integrals given by 
  
$$
P(\beta_P|D) \propto \int d\sigma \int d\beta_M \int d\beta_N \int d\alpha \ P(D | \alpha,\beta_N,\beta_P, \beta_M,\sigma)P(\beta_P)P(\beta_M)P(\beta_N)P(\sigma),\\
P(\beta_M|D) \propto \int d\sigma \int d\beta_P \int d\beta_N \int d\alpha \ P(D | \alpha,\beta_N,\beta_P, \beta_M,\sigma)P(\beta_P)P(\beta_M)P(\beta_N)P(\sigma),\\
P(\beta_N|D) \propto \int d\sigma \int d\beta_P \int d\beta_N \int d\alpha \ P(D | \alpha,\beta_N,\beta_P, \beta_M,\sigma)P(\beta_P)P(\beta_M)P(\beta_N)P(\sigma), \\
$$


and the function that determines the likelyhood of the data given the model parameters is


$$
\\ P(D | \alpha,\beta_N,\beta_P, \beta_M,\sigma) = \text{Exp}\left( - \frac{1}{2\sigma^2}\chi^2 \right), \\
$$

$$\sigma$$ is the width of the distribution and $$\chi^2$$ is simply the least square residuals of the linear model and data.

The priors of the parameters $$P(\beta_{N})$$, $$P(\beta_{M})$$, $$P(\beta_{P})$$, $$P(\alpha)$$ have been chosen to be uniformly 
distributed in the range (-100,100).  The prior $$P(\sigma)$$ was chosen to be a Jeffreys prior. The results from this analysis 
are not expected to change too much for different priors, so we do not change this. 
We calculate the posterior distributions for all 5 car brands.  The distribution for the coefficient 
$$\beta_P$$ for Tesla is shown below,

<figure>
    <img src="/assets/img/tesla_bP_coeff.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: A summary of the correlations values found over the investigated time
    period
    </figcaption>
</figure>

This figure shows the medeian values as well as the 68%, and 95% confidence intervals. 
These confidence intervals demonstrate the advantage of using the Bayesian linear regression. 


From this plot we see that the mean value of the fit is a positive value that excludes zero at the 68% confidence level,
 but clearly we don't have enough statistical power to exclude 0 at the 95% confidence level.

Next, we would like to calculate the posterior distributions of the correlation coefficients between the sentiment signals and the returns of the stock.

A summary of the correlations that we found is given below,

<figure>
    <img src="/assets/img/correlation_diagram.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: A summary of the correlation values found over the investigated time
    period. Green ovals indicate positive association, red is negative, orange circle indicates no correlation.
    </figcaption>
</figure>


The main findings of this table are:

* For all vehicle brands that we looked at, the median value of $$\beta_M$$, representing the sensititivity of the market return to
the stock return was always positive and seemed to be the strongest effect in the linear model.

* The stock returns for __Tesla__ are __positively correlated__ with Negative twitter sentiments  
* The stock returns for __Toyota and Porche__ are __positively correlated__ with Positive twitter sentiments 

* The stock returns for __Tesla and Ford__ are __negatively correlated__ with Positive twitter sentiments  
* The stock returns for __Mercedes__ are __negatively correlated__ with Negative twitter sentiments

In future work, it would be good to try to understand the underlying reasons why the sentiments are correlated the way 
we found them here. It is counter intuitive that positive tweet sentiments are anti-correlated with the return of the 
automotive stocks for some companies.  


## Posteriors of the correlations

The results of the last section give us the general "sign" of the correlations, but not the numerical values 
of their strengths. It will be our next task to compute the posterior distributions of these correlations. The 
pearsons coefficient that we are interested in computing between time series $$x$$ and $$y$$ is

$$
\rho(x,y) = \frac{\sum\limits_{i}^N (x_i-\bar{x})(y_i-\bar{y})}{(N-1) s_x s_y}, \\
s^2_{x} = \frac{1}{N-1}\sum\limits_{i}^{N} (x_i-\bar{x})^2, \\
s^2_{y} = \frac{1}{N-1}\sum\limits_{i}^{N} (y_i-\bar{y})^2. 
$$

where $$N$$ is the number of data points. Here $$\bar{x}$$ and $$\bar{y}$$ are the mean values of the time series.

<figure>
    <img src="/assets/img/correlation_posterior.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: The posterior distribution of the Perasons correlation between positive sentiment 
    and the stock return for Tesla. 
    </figcaption>
</figure>

<br>
Finally we summarize the median values of the correlations that we found which numerically quantify the relations that 
we found in the previous section.
<br>

<figure>
    <img src="/assets/img/correlation_values.png" width="70%" class="center"
         alt="Elephant at sunset">
    <figcaption class="center">Fig. 4: A summary of the Pearsons correlation values found over the investigated time
    period.
    </figcaption>
</figure>

# Conclusions and further work
In this article, we focused more on the mathematical details and the data analysis methodology that was built with python that allowed us
 to carry out the end-to-end data science project. The main limitation of our work was the small data set that was used to compute the correlations. 
However this proof-of-principle project produced potentially interesting results. In the next upcoming blog post, we will use a much larger data set, 
with several months worth of data to more rigorously study the model proposed here. In that upcoming post we will study in detail the following effects:
* __How does the sentiment correlations change as a function of time?__  
* __What are the results of using a sliding window approach for the correlations?__  
* __Does changing the normalization method change the correlations? For example using a min-max scaler as opposed to the Z-transformed data?__  
* __What are the results when we include/remove outliers in the data set?__  

### __Hopefully we will be able to satisfy our curiosity in the next post and we wont have to cover so much math, stay tuned!__ 
