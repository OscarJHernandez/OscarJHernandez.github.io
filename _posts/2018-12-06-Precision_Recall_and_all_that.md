---
layout: resume
title:  "Precision calculus and all that"
date:   2018-12-06 00:00:00
categories: jekyll
---

# Precision, Recall and all that  

Let us now consider a binary classification task. Our training data is denoted as the set  $${\bf T}$$, where

$$
{\bf T} = \lbrace (x_i,y_i)| i=1,...,N \rbrace
$$ 



Our data vector $${\bf y}$$ can have two outcomes, $$y_k= 0 \text{ or } 1$$. 
Let us call the outcome with a +1 value, be the positive value while the 0 is called the negative value.

# Table of Contents
1. [Accuracy](#accuracy)
2. [Precision](#precision)
3. [Recall](#recall)
4. [The F Metric](#the-f-metric)
5. [The link between accuracy, recall and precision](#connection_between_accuracy_,_recall_and_precision)

## Definitions

For our binary classifier model trained on $$T$$ is denoted as $$\hat{g}_T.$$ The number of positive samples in $$T$$ is 
$$N_p$$, while the number of negative samples is $$N_n$$, then we define the following

$$
N = N_p+N_n \\
T_p = \text{ True Positives} \\
T_n = \text{ True Negatives} \\
F_p = \text{ False Positives} \\
F_n = \text{ False Negatives} \\
N = T_p+T_n+F_p+F_n.
$$


Furthermore, we denote

$$
\hat{N}_p = \text{ Number of positive predictions by classifier}, \\
\hat{N}_n = \text{ Number of negative predictions by classifier}, \\
\hat{N}_p+\hat{N}_n = N.
$$

With these definitions we have that the confusion matrix is defined as

$$
C= \begin{bmatrix}
T_p & F_p\\
F_n & T_n
\end{bmatrix} 
$$

when we divide $$C$$ by the number of samples, then we can give the confusion matrix a probabilistic interpretation

$$
\mathcal{C}= \begin{bmatrix}
P({\bf \hat{y}}={\bf 1}|{\bf y}={\bf 1}) & P({\bf \hat{y}}={\bf 1}|{\bf y}={\bf 0})\\
 P({\bf \hat{y}}={\bf 0}|{\bf y}={\bf 1}) & P({\bf \hat{y}}={\bf 0}|{\bf y}={\bf 0})
\end{bmatrix} 
$$

In the special case that our classifier doesn't make any mistakes, the false negatives and positives are zero, therefore 

$$
C_{\text{Perfect}}= \begin{bmatrix}
T_p & 0\\
0 & T_n
\end{bmatrix} 
$$

In the other extreme where the classifier doesn't make any correct classifications we have

$$
\mathcal{C}_{\text{Terrible}}= \begin{bmatrix}
0 & F_p\\
F_n & 0
\end{bmatrix} 
$$

And in the case where the classifier is totally random,

$$
\mathcal{C}_{\text{Random}}= \begin{bmatrix}
\frac{1}{4} & \frac{1}{4}\\
\frac{1}{4} & \frac{1}{4}
\end{bmatrix} 
$$

## Accuracy
How well did the classifier get the correct labels.

$$
a = P({\bf y}={\bf \hat{y}}) \\
= P({\bf y}={\bf \hat{y}}, {\bf y}=1)+P({\bf y}={\bf \hat{y}}, {\bf y}=0)\\
= \left( \frac{T_p}{N} \right) + \left( \frac{T_n}{N} \right) \\
= \frac{T_p+T_n}{N}
$$

$$
 0 \leq a \leq 1
$$

* a __perfect__ classifier would have $$a=1$$.  
* a __terrible__ classifier has $$a=0$$  
* a __random__ classifier has $$a=\frac{1}{2}$$


## Precision

Of the samples, $$\hat{N}_p$$, that the classifier thought were positive, how many are actually correct ?

$$
p = P({\bf y}={\bf 1}| {\bf \hat{y}} = {\bf 1}) \\
  = \frac{T_p}{T_p+F_p}
$$

$$0 \leq p \leq 1$$

* a __perfect__ classifier would have $$p=1$$.  
* a __terrible__ classifier has $$p=0$$  
* a __random__ classifier has $$p=\frac{1}{2}$$


## Recall 
Recall, is the metric

$$
r = P( {\bf \hat{y}}={\bf 1} | {\bf y}={\bf 1}) \\
  = \frac{T_p}{T_p+F_n}
$$

$$0 \leq r  \leq 1$$

* a __perfect__ classifier would have $$r=1$$.  
* a __terrible__ classifier has $$r=0$$  
* a __random__ classifier has $$r=\frac{1}{2}$$

## The F Metric
The $$F_\beta$$ metric is the following function of precision and recall.

$$
F_{\beta} = (1+\beta^2) \frac{p r}{\beta^2p+r}
$$

clearly

$$
0 \leq F_\beta \leq 1
$$

In the case where $$\beta$$=1, then $$F_1$$ is the harmonic mean of the precision and recall

$$
F_1 = \frac{2pr}{r+p}
$$

* a __perfect__ classifier would have $$F_\beta=1$$,  
* a __terrible__ classifier has $$F_\beta=0$$,  
* a __random__ classifier has $$F_\beta=\frac{1}{2}$$.

## ROC curve

Let us now consider the density given by,

<center>
<img src="/assets/img/binary_classifier.png"
     alt="Markdown Monster icon"
     width ="50%"
     height = "50%"/>
</center>
<center>
$${\bf \text{Figure 1: }}\text{The probability distribution of the true density }T_p(x) \text{ and the true negative distribution } T_n(x)$$. 
</center>

$$
\rho(x) = T_p(t)F_p(t)
$$


# Connection between Accuracy, Recall and Precision
Having written down Recall and precision as conditional probabilities, it is much 

$$
P({\bf \hat{y}}={\bf 1}) = \frac{\hat{N}_p}{N} = \frac{T_p+F_p}{N} \\
P({\bf \hat{y}}={\bf 0}) = \frac{\hat{N}_n}{N} = \frac{T_n+F_n}{N} \\
\text{ }\\
P({\bf y}={\bf 1}) =  \frac{N_p}{N} =  \frac{T_p+F_n}{N} \\
P({\bf y}={\bf 0}) =  \frac{N_n}{N} = \frac{T_n+F_p}{N} 
$$


By __Bayes theorem__ we have that

$$
P({\bf y}={\bf 1}| {\bf \hat{y}} = {\bf 1}) = \frac{P({\bf \hat{y}} = {\bf 1}|{\bf y}={\bf 1})P({\bf y}={\bf 1})}{P({\bf \hat{y}}={\bf 1})} \\
= P({\bf \hat{y}} = {\bf 1}|{\bf y}={\bf 1}) \frac{T_p+F_n}{T_p+F_p} \\
= P({\bf \hat{y}} = {\bf 1}|{\bf y}={\bf 1})\left(\frac{1+\frac{F_n}{T_p}}{1+\frac{F_p}{T_p}} \right)
$$

In otherwords, we find that

$$
p = r \left(\frac{1+\frac{F_n}{T_p}}{1+\frac{F_p}{T_p}} \right)
$$



### References
[Stack-Overflow](https://datascience.stackexchange.com/questions/18099/interpretation-of-recall-as-a-conditional-probability-p-x-x)
