---
layout: post
title: Deep Learning
excerpt_separator: <!--more-->
---

<div class="message">
  Howdy! This post is under construction.
</div>

 This series of posts on Deep learning is my notes on Deep learning from the course [6.S191 Introduction to Deep Learning](http://introtodeeplearning.com). It is meant to be more of a personal notes so if you find something less descriptive please follow this [link](http://introtodeeplearning.com).

 <!--more-->

## The Perceptron


### Structural building blocks

A Perceptron is the fundamental building block of neural networks. The idea of Perceptron's can be traced back to the functioning of a **Neuron** in animals, i.e. when activated it fires a pulse of current. Similarly, a Perceptron usually takes input, multiplies it with a weight and squashes it with a activation function, represented via the following equation.

$$
  \begin{align*}
    \hat{y} =  g \left(W_0 + \sum_{i=1}^n w_ix_i \right)
  \end{align*}
$$

$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$

### Nonlinear activation functions

text here

## Neural Networks

text here

### Stacking Perceptrons to form neural networks

text here

### Optimization through backpropagation

text here


## Training in Practice

text here

### Adaptive learning

text here

### Batching

text here

### Regularization

text herex
