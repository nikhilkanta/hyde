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

A Perceptron is the fundamental building block of neural networks. The idea of Perceptron's can be traced back to the functioning of a **Neuron** in animals, i.e. when activated it fires a pulse of current. Similarly, a Perceptron usually takes input, multiplies it with a weight and squashes it with a activation function, which can be represented via the following equation.

$$
  \begin{align*}
    \hat{y} =  g \left(w_0 + \sum_{i=1}^n w_ix_i \right)
  \end{align*}
$$

Where $$\hat{y}$$ is the output, $$g(\cdot)$$ is the activation function, $$n$$ is the number of inputs,  $$w_i$$ are the respective weights for the input $$x_i$$ and $$w_0$$ is the bias.

![Perceptron](assets\posts\Deep_learning\perceptron.png)
*fig: A Perceptron*

The vectorial representation of the above equation is given by

$$
\begin{align*}
  \hat{y} =  g \left(w_0 + \bf{X^TW} \right)
  \end{align*}
$$

The activation function $$g(\cdot)$$ used are usually a non-linear activation function. A example of a non-linear activation function is the sigmoid function which is given by.

$$
  \begin{align*}
    g(z) =  \sigma(z) = \frac{1}{1 + e^{-z}}
  \end{align*}
$$

![sigmid_function](assets\posts\Deep_learning\sigmoid_function.png)
*fig: A sigmoid activation function*

### Nonlinear activation functions

Why a Non-linear activation function? The whole purpose of activation function is tp introduce non-linearities to the network. A linear activation function produces a linear decision boundary. As we know all the real world systems are non-linear it makes much sense to use a non-linear activation function.

Here are some activation functions with their use code in Tensorflow.

![activaion_functions](assets\posts\Deep_learning\activation_function.png)
*fig: Common activation functions*


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
