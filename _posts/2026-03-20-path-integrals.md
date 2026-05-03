---
layout: post
title: "Path Integrals: From Quantum Mechanics to Statistical Field Theory"
date: 2026-03-20
category: Mathematics
description: A self-contained introduction to Feynman's path integral formulation, with connections to statistical mechanics.
---

## Introduction

The path integral formulation of quantum mechanics, due to Feynman (1948), provides an alternative to the operator formalism of Heisenberg and the wave equation of Schrödinger. It expresses the quantum mechanical amplitude for a particle to travel from point $x_a$ at time $t_a$ to $x_b$ at time $t_b$ as a sum over all possible paths.

## The Propagator

In standard quantum mechanics, the propagator (or kernel) is:

$$
K(x_b, t_b; x_a, t_a) = \langle x_b | e^{-iH(t_b - t_a)/\hbar} | x_a \rangle
$$

Feynman's insight was to write this as:

$$
K(x_b, t_b; x_a, t_a) = \int \mathcal{D}[x(t)] \, \exp\!\left(\frac{i}{\hbar} S[x(t)]\right)
$$

where $S[x(t)]$ is the classical action:

$$
S[x(t)] = \int_{t_a}^{t_b} \mathcal{L}\!\left(x, \dot{x}, t\right) dt
$$

and $\int \mathcal{D}[x(t)]$ denotes a sum (integral) over all paths $x(t)$ satisfying $x(t_a) = x_a$ and $x(t_b) = x_b$.

## Discretized Form

To make this precise, divide $[t_a, t_b]$ into $N$ slices of width $\epsilon = (t_b - t_a)/N$:

$$
K = \lim_{N \to \infty} \left(\frac{m}{2\pi i \hbar \epsilon}\right)^{N/2} \int_{-\infty}^{\infty} \prod_{k=1}^{N-1} dx_k \; \exp\!\left(\frac{i}{\hbar} \sum_{k=0}^{N-1} \frac{m(x_{k+1} - x_k)^2}{2\epsilon}\right)
$$

## Connection to Statistical Mechanics

Under the Wick rotation $t \to -i\tau$ (Euclidean time), the Minkowski path integral becomes:

$$
K_E = \int \mathcal{D}[x(\tau)] \, \exp\!\left(-\frac{1}{\hbar} S_E[x(\tau)]\right)
$$

This is formally identical to the partition function in statistical mechanics:

$$
Z = \int \mathcal{D}[\phi] \, e^{-\beta H[\phi]}
$$

with the identification $\beta \leftrightarrow 1/\hbar$ and periodic boundary conditions $x(0) = x(\beta\hbar)$ encoding the trace over states.

## Gaussian Path Integrals

The simplest case is the harmonic oscillator, $V(x) = \frac{1}{2}m\omega^2 x^2$. The action is quadratic, and the path integral is Gaussian:

$$
K = \sqrt{\frac{m\omega}{2\pi i \hbar \sin(\omega T)}} \exp\!\left(\frac{im\omega}{2\hbar\sin(\omega T)}\left[(x_a^2 + x_b^2)\cos(\omega T) - 2x_ax_b\right]\right)
$$

where $T = t_b - t_a$.

## Further Reading

- Feynman & Hibbs, *Quantum Mechanics and Path Integrals* (1965)
- Zinn-Justin, *Quantum Field Theory and Critical Phenomena*
- Kleinert, *Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets*
