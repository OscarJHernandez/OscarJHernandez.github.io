---
layout: blog
title: "Path Integrals"
date:   2018-08-08 13:50:39
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


In physics, we often deal with 


# Table of Contents
1. []
2. [Brownian Path Integral](#brownian-path-integral)
3. [Brownian Path Integral](#brownian-path-integral)
4. [Bayesian fit of Brownian Motion](#precision)



# Stochastic Processes


# The brownian walk as a path integral

The stochastic integral that we want to model is 
$$
X_{T} = X_{T-\Delta T}+\eps,
$$
where 
$$
\eps \sim \mathcal{N}(0,\sigma^2 \Delta T),
$$
and $$\matcal{N}(\mu,\sigma^2)$$ is a normal distribution with mean $$\mu$$ and variance $$\sigma^2$$. 

The question that we want to answer is: "Given a starting position at X_0 at T=0, what is "
$$
P(X_T) = \int dX_{T-\Delta T} \int d\eps P(X_T,X_{T-\Delta T},\eps)
$$



