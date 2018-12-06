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


## Definitions

For our binary classifier model $\hat{g}_T$

$$
C= \begin{bmatrix}
T_p & F_p\\
F_p & T_n
\end{bmatrix} 
$$

$$
T_p = \text{ True Positives} \\
T_n = \text{ True Negatives} \\
F_p = \text{ False Positives} \\
F_n = \text{ False Negatives}
$$

## Accuracy
How well did the classifier get the correct labels.

$$
a = P({\bf y}={\bf \hat{y}}) \\
= P({\bf y}={\bf \hat{y}}, {\bf y}=1)+P({\bf y}={\bf \hat{y}}, {\bf y}=0)\\
= \left( \frac{T_p}{T_p+T_n+F_p+F_n} \right) + \left( \frac{T_n}{T_p+T_n+F_p+F_n} \right) \\
= \frac{T_p+T_n}{T_p+T_n+F_p+F_n}
$$

$$
 0 \leq a \leq 1
$$

a perfect classifier would have $$a=1$$.


## Precision

How 

$$
p = P({\bf y}={\bf 1}| {\bf \hat{y}} = {\bf 1}) \\
  = \frac{T_p}{T_p+F_p}
$$

$$0 \leq p \leq 1$$

## Recall 
Recall, is the metric

$$
r = P( {\bf \hat{y}}={\bf 1} | {\bf y}={\bf 1}) \\
  = \frac{T_p}{T_p+F_n}
$$

$$0 \leq r  \leq 1$$

# Connection between Accuracy, Recall and Precision
Having written down Recall and precision as conditional probabilities, it is much 

$$
P({\bf \hat{y}}={\bf 1}) = \frac{T_p+F_p}{T_p+T_n+F_p+F_n} \\
P({\bf \hat{y}}={\bf 0}) = \frac{T_n+F_n}{T_p+T_n+F_p+F_n} \\
\text{ }\\
P({\bf y}={\bf 1}) = \frac{T_p+F_n}{T_p+T_n+F_p+F_n} \\
P({\bf y}={\bf 0}) = \frac{T_n+F_p}{T_p+T_n+F_p+F_n} 
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
