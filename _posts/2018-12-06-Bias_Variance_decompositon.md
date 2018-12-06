---
layout: resume
title:  "Bias Variance Decomposition"
date:   2018-12-06 00:00:00
categories: jekyll
---


# Bias-Variance Decomposition

In this note, I will derive the Bias-Variance decomposition following Ref. [1](),[2]().

Let us suppose there exists the real function $$g({\bf x})$$ that generates our data $$y$$ with additive noise, so 

$$
{\bf y} = g({\bf X})+{\bf \epsilon}.
$$

Let $$T$$ be the set of training data

$$ T=\lbrace (x_i,y_i) | i=0,...,N_T \rbrace $$  

And suppose we train a model $$\hat{g}_T$$ with this set. The cost function $$J({\bf X},\hat{g}_T)$$ that we consider here is given by  

$$
J({\bf X},\hat{g}_T) = \left( {\bf y}- {\bf \hat{g}_T(X)} \right)^2
$$

where $${\bf y}=(y_1,...,y_N)$$ and $${\bf \hat{g}_T(X)}=(\hat{g}_T({\bf x}_1),...,\hat{g}_T({\bf x}_N))$$.


The expectation value of $$J$$, denoted by $$\langle J \rangle$$, will be given by    

$$
\langle J \rangle = \langle \left( {\bf y}-{\bf g(x)}+{\bf g(x)}- {\bf \hat{g}_T(x)} \right)^2 \rangle \\
= \langle \left( {\bf y}-{\bf g(X)}+{\bf g(X)}- {\bf \hat{g}_T(x)} \right)^2 \rangle \\
=  \langle \left( {\bf y}-{\bf g(X)} \right)^2 \rangle + \langle \left( {\bf g(X)}- {\bf \hat{g}_T(x)} \right)^2 \rangle + 2 \langle \left( {\bf y}- {\bf \hat{g}_T(x)} \right)\rangle \langle \left( {\bf g(X)}- {\bf \hat{g}_T(x)} \right) \rangle \\
=  \langle \left( {\bf y}-{\bf g(X)} \right)^2 \rangle + \langle \left( {\bf g(X)}- {\bf \hat{g}_T(x)} \right)^2 \rangle 
$$





### References
 
