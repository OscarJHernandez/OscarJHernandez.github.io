---
layout: resume
title:  "Precision calculus and all that"
date:   2018-12-06 00:00:00
categories: jekyll
---

Let us now consider a binary classification task. Our data vector $${\bf y}$$ can have two outcomes, $$y_k=(0,1)$$. 
Let us call the outcome with a +1 value, be the positive value while the 0 is called the negative value.

# Precision

$$
P = P({\bf y}={\bf 1}| {\bf \hat{y}} = {\bf 1}) \\
  = \frac{T_p}{T_p+F_p}
$$


# Recall 
Recall, is the metric

$$
R = P( {\bf \hat{y}}={\bf 1} | {\bf y}={\bf 1}) \\
  = \frac{T_p}{T_p+F_n}
$$


# Connection between Recall and Precision
Having written down Recall and precision as conditional probabilities, it is much 

$$
P({\bf \hat{y}}={\bf 1}) = \frac{T_p+F_p}{T_p+T_n+F_p+F_n} \\
P({\bf y}={\bf 1}) = \frac{T_p+F_n}{T_p+T_n+F_p+F_n}
$$



### References
[Stack-Overflow](https://datascience.stackexchange.com/questions/18099/interpretation-of-recall-as-a-conditional-probability-p-x-x)
