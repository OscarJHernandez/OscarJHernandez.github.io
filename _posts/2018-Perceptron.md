---
layout: resume
title:  "The Perceptron Notes"
date:   2018-11-20 00:00:00
categories: jekyll
---


# The Perceptron

$$
D = \lbrace ({\bf X_i},y_i) | i = 1,...,N \rbrace
$$
the vector $${\bf X}$$ is the feature vector of dimension $$M$$.


By convention, we will define $$X_{0}=1$$
$$
\hat{g}_{T}({\bf X})= \begin{matrix} 1, \text{ if } {\bf W} \cdot {\bf X} \geq 0 \\ 0 \text{ otherwise } \end{matrix}
$$

From linear algebra we know that the equation
$$
{\bf W} \cdot {\bf X} = 0
$$
defines a plane with a normal vector given by $$\hat{n}= \left( W_1,..., W_N \right)$$. When 
$${\bf W} \cdot {\bf X} > 0$$, then the point $${\bf X}$$ lies above the plane, while when $${\bf W} \cdot {\bf X} < 0$$
the point lies below.

Let us also recall that the distance from this plane to the point $$P = (1,p_1,...,p_M)$$ is given by

$$
d = \frac{|{\bf P}\cdot {\bf W}|}{|{\bf W}|}
$$


## Training Algorithm

We want to adapt the set of weights $${\bf W}$$ so that it minimizes the classification error. Let us 
take the training data to be $$D_T = \lbrace ({\bf X}_i,y_i) | i=1...M_T \rbrace$$

0. Normalize all training examples $$D_T$$, initialize $$\epsilon > 0$$  
1. Randomly generate the vector $${\bf W}$$  
2. Identify all of the misclassified points $$ K= \lbrace ({\bf X}_i,y_i) | i =1,...,M_K \rbrace$$
3. 
$$
W_t \rightarrow W_{t-1}+\epsilon{\bf X} \\
W_t \rightarrow W_{t-1}-\epsilon{\bf X} 
$$
### Python Implementation

## Bayesian Training Routine

Now that we have 

0. Optimize 

$$
\begin{align}
P({\bf W}|D) &\proto P(D|{\bf W})P({\bf W}) \\
P(D|{\bf W}) &= \int d\sigma \text{Exp}\left( -\frac{1}{2\sigma^2} \sum_{i=1}^{K} d^2_{i} \right)P() \\
d_{i} &= \frac{{\bf X}\cdot {\bf W}}{|{\bf W}|}
\end{align}
$$

### Python Implementation
