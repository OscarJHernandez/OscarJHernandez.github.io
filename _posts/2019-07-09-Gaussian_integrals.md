---
layout: blog
title: "Gaussian Integrals"
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


# Gaussian Integrals

The following formulas are useful for derivations involving Gaussian integrals.

## General integrals

$$
\int\limits_{-\infty}^{\infty} dx \ \text{Exp}\left( -ax^2+bx+c \right) = \sqrt{\frac{\pi}{a}} \text{Exp}\left( \frac{b^2}{4a}+c \right)
$$

$$
\int\limits_{-\infty}^{\infty} dx \ \frac{1}{\sqrt{2\pi\sigma^2}}\text{Exp}\left( -\frac{1}{2\sigma^2}\left(x-\mu \right)^2+\lambda x \right) = \text{Exp}\left( \frac{\sigma^2}{2}\lambda^2+\lambda \mu \right)
$$

## Multivariate Gaussian Integrals

$$
\int dx_1 \ldots dx_n \text{Exp}\left( -\frac{1}{2} \vec{x}^T A \vec{x} + \vec{J}^T \cdot \vec{x} \right) = \sqrt{\frac{(2\pi)^n}{|\Sigma|}} \text{Exp}\left( -\frac{1}{2} \vec{J}^{T} A^{-1} \vec{J} \right)
$$

$$
\int dx_1 \ldots dx_n \int dk_1 \ldots dk_n \text{Exp}\left( -i\vec{k}^T A \vec{x} + i\vec{k}^T \vec{J}_1 ++ i\vec{J}_2^T \vec{x} \right) \propto \text{Exp}\left( \vec{J}_2^{T} A^{-1} \vec{J}_1 \right)
$$

## Representations of Delta Function

$$
\delta(x) = \frac{1}{2\pi} \int\limits_{-\infty}^{\infty} dk \ e^{ikx}
$$

In three dimensions  

$$
\delta(\vec{x}) = \frac{1}{(2\pi)^3} \int\limits_{-\infty}^{\infty} d^3k \ e^{i\vec{k}\cdot \vec{x}}.  
$$



The following property of the delta function is also very useful
$$
\delta(x) = \delta(y)\left[1+\sum_{n=1}^{\infty} \left( - \frac{\partial }{\partial x} \right)^n \int\limits dy \ \left(x-y\right)^n \right] 
$$



## Products of Gaussians

$$
\int dx_1 \frac{1}{\sqrt{2\pi \sigma^2}} \text{Exp}\left(\frac{1}{2\sigma^2}\left(x_2-b x_1-u_2 \right)^2 \right)  \frac{1}{\sqrt{2\pi \sigma^2}} \text{Exp}\left( \frac{1}{2\sigma^2}\left(x_1 -b x_0 -u_1 \right)^2 \right) = \frac{1}{\sqrt{2\pi\sigma^2(1+b^2)}}\text{Exp}\left( -\frac{(x_2-b^2 x_0 -b u_1-u_2)^2}{2\sigma^2(1+b^2)} \right)
$$



$$
\int dx_1 \int dx_2  \frac{1}{\sqrt{2\pi \sigma^2}} \text{Exp}\left(\frac{1}{2\sigma^2}\left(x_3-b x_2-u_3 \right)^2 \right) \frac{1}{\sqrt{2\pi \sigma^2}} \text{Exp}\left(\frac{1}{2\sigma^2}\left(x_2-b x_1-u_2 \right)^2 \right)  \frac{1}{\sqrt{2\pi \sigma^2}} \text{Exp}\left( \frac{1}{2\sigma^2}\left(x_1 -b x_0 -u_1 \right)^2 \right) =\\
  \frac{1}{\sqrt{2\pi \sigma^2(1+b^2+b^4)}}\text{Exp}\left( -\frac{(x_3-b^3 x_0-b^2 u_1 -b u_2-u_3)^2}{2\sigma^2(1+b^2+b^4)} \right)
$$


## Bivariate Gaussian

The following notation is useful for the case of the Bivariate Gaussian distribution  


$$
\vec{\mu} = \begin{bmatrix} 
\mu_1 \\
\mu_2
\end{bmatrix}, \\

\Sigma = \begin{bmatrix}
\sigma^2_1 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{bmatrix}\\

\vec{x} = \begin{bmatrix} 
x_1 \\
x_2
\end{bmatrix}, \\

$$

where $$\vec{\mu}$$ is the vector of mean values, $$\vec{x}$$ is the vector of variables and $$\Sigma$$ is the covariance matrix. 
The joint probability distribution of $$x_1$$ and $$x_2$$ is  

$$
P(x_1,x_2) = \frac{1}{\sqrt{(2\pi)^2|\Sigma|}} \text{Exp}\left[ \left(\vec{\mu}-\vec{x}\right)^{T}\Sigma^{-1} \left(\vec{\mu}-\vec{x} \right) \right]
$$


## Conditional Probabilities

Here we look at the conditional probability distribution of the multivariate Gaussian distributions. For the bivariate Gaussian the
conditional probability is

$$
P(x_1| x_2) = \frac{1}{\sqrt{2\pi(1-\rho^2)\sigma^2_1}} \text{Exp}\left( -\frac{1}{2(1-\rho^2)\sigma^2_1}\left(x_1 -\mu_1 - \frac{\sigma_1}{\sigma_2}\rho(x_2-\mu_2) \right)^2 \right),
$$

while the general case is

$$
P(\vec{x}| \vec{y}) = \frac{1}{\sqrt{ (2\pi)^k |{\bf \hat{\Sigma} }| }} \text{Exp}\left( - (\vec{x}-\hat{\mu})^T  \hat{{\bf \Sigma}}^{-1} (\vec{x}-\hat{\mu})  \right)
$$


where we have that

$$
\hat{\mu} = \vec{\mu}_1+\Sigma_{12}\Sigma^{-1}_{22}(\vec{y}-\vec{\mu}_2)
$$


and   


$$
\hat{\Sigma} = {\bf \Sigma}_{11} - {\bf \Sigma}_{12}{\bf \Sigma}^{-1}_{22}{\bf \Sigma}_{21}
$$

with 

$$
\hat{\mu} = \begin{bmatrix} 
\mu_1 \\
\mu_2
\end{bmatrix}, \\

{\bf \Sigma} = \begin{bmatrix}
{\bf \Sigma_{11}} & {\bf \Sigma_{12}} \\
{\bf \Sigma_{21}} & {\bf \Sigma_{22}}
\end{bmatrix}\\

\hat{x} = \begin{bmatrix} 
\vec{x} \\
\vec{y}
\end{bmatrix}, 
$$
