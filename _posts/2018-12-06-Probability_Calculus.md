---
layout: resume
title:  "Probability Calculus Notes"
date:   2018-12-06 00:00:00
categories: jekyll
---


# Definitions and useful formulas

### Joint Conditional Probability
The probability of conditions $$A$$ and $$B$$ occuring at the same time

$$
P(A,B) \equiv P(A \text{ and } B)
$$

## Addition Rule

The probability of $$A$$ or $$B$$ occuring is given by

$$
P(A \cup B) = P(A)+P(B)-P(A,B)
$$

### Conditional Probability
The conditional Probability, of an event $$A$$ given a condition $$B$$, denoted $$P(A|B)$$ is defined as

$$
P(A|B) = \frac{P(A,B)}{P(B)}
$$


### Chain Rule

$$
P(A,B) = P(A|B) P(B) 
$$

### Marginalization

$$
\begin{align}
P(A) &= \sum\limits_{i=0}^N P(A,B_i) \\
     &= \sum\limits_{i=0}^N P(A|B_i)P(B_i)
\end{align}
$$

In the continuous case we have

$$
\begin{align}
P(A) &= \int d\theta \ P(A,\theta) \\
 &= \int d\theta \ P(A|\theta)P(\theta)
\end{align}
$$


### Bayes Theorem

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$


