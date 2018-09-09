---
layout: resume
title: Losing weight the Bayesian Way
date:   2018-06-04 13:50:39
categories: Bayesian
---

If you've been thinking of going on a diet, like me, then perhaps you're interested on quantifying the impact that your new
diet is having on your weight. Collecting data requires you to have a scale and to measure yourself consistently for several weeks 
in order to have enough data to analyse. Once you have the weight vs time data, you could just plot it using any software
and fit a line to it. The slope of the line will tell you roughly how much you're gaining/losing, but usually such software does not
tell you about the uncertainty of the determined parameters.    
 
When I'm doing this kind of analysis I'm always interested in generating my weekly weight trend prediction with 95% confidence
intervals. So the usual excel software is very lacking for me. Generating these confidence intervals can be a bit tricky, but
one way to do it would be to carry out a Bayesian analysis of the fit.  

In this blog post, we will explore the Bayesian way of carrying out a linear fit on sample weight data to determine
(with confidence intervals!) how much weight you're gaining or losing with some sample code. Hopefully we will all learn some
Bayesian theory in the process and reach our weight goals. Now lets get our hands dirty with some theory.  


# Theory
The main idea is that we have a set of data points ![image](http://latex.codecogs.com/gif.latex?D=\lbrace&space;x_i,y_i&space;|&space;i=0,..,N&space;\rbrace) and we want to 
determine the parameters of the model ![image](http://latex.codecogs.com/gif.latex?M) that will best descibribe the set ![image](http://latex.codecogs.com/gif.latex?D). We begin by writing out 


![equation](http://latex.codecogs.com/gif.latex?P(a|D)&space;=&space;\int&space;db&space;\&space;P(a,b|&space;D))  

![equation]

## Sampling the Posterior (Markov Chain Monte Carlo)  
Now, we have written down the likelyhood function that we want to sample. We will use the pythong library emcee, which is an 
excellent package for doing MCMC.


# So how much weight will I loose?
Now, we are interested in the slope $a$ of the linear fit which will tell us approximately how much weight you will be expected
to loose in a given week.



