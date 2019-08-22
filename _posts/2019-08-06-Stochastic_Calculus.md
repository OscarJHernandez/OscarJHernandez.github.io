---
layout: blog
title: "Random walks, partial differential equations and path integrals"
date:   2019-07-09 00:00:00
categories: other
image: /assets/img/EDS3_figure1.png
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


# Introduction 

In this post, we will explore stochastic calculus

$$
X_{T} = X_{T-1}+ \Delta t \cdot f(X_{T-1},T-1) + \sqrt{\Delta t} \sigma(X_{T-1},{T-1}) \cdot z_{T-1} ,
$$
where $z_T$ is a random variable with a specified probability distribution $P(z_T)$. We assume that 
the values $z_T$ are each independent and identically distributed. 
$$
P(X_T,T) = \int dX_{T-1} \ \int dr \ P(X_{T},T|X_{T-1},T-1,r)P(X_{T-1},T-1,r) \\
= \int dX_{T-1} \ \int dr \ P(X_{T},T|X_{T-1},T-1,r)P(X_{T-1},T-1)P(r),
$$
where we have assumed that the random variable $r$ is independent of $X_{T-1}$ and $T-1$.
Using the constraint that  
$$
P(X_{T},T|X_{T-1},T-1,r) = \delta(X_{T} -X_{T-1}- \Delta t \cdot f(X_{T-1},T-1) + \sqrt{\Delta t} \sigma(X_{T-1},T-1) \cdot r),
$$
the marginalization of the integral over $r$ results in
$$
P(X_T,T) = \int dX_{T-1}  \ P(X_{T-1},T-1)P(r = \frac{\left((X_{T} -X_{T-1}- \Delta t \cdot f(X_{T-1},T-1)  \right)}{\Delta t \cdot \sigma(X_{T-1},T-1)}).
$$

# Path integrals
Now that we have an integral expression relating the previous probability distributions to the current time, we 
can use the expression recurrently

$$
P(X_T,T) = \int dX_{T-1}  \ \int dX_{T-1}  \ \cdots  P(r_T)P(r_{T-1}).
$$



# The Fokker-Planck equation

# Geometric Brownian motion and the Stock Market



