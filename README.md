# Custom Autograd Engine and Neural Network

## Overview
This repository contains a from-scratch implementation of an automatic differentiation (autograd) engine and a lightweight neural network library. The project demonstrates the mathematical intuition and inner workings of deep learning frameworks by implementing the forward and backward passes purely using matrix operations and the chain rule.

## Project Transparency
To clearly distinguish between custom logic and external utilities:
* **Core Engine:** The entire computational graph, topological sorting, gradient tracking, layer architectures, and optimizers are built **100% from scratch** utilizing only standard mathematical operations in `numpy`.
* **External Libraries:** `scikit-learn` is utilized strictly for fetching and splitting the standard benchmark datasets (MNIST and Two Moons). 

## Features
* **Custom Tensor Class:** A `Tensor` object that tracks data, gradients, and the chain of operations (children)[cite: 1].
* **Automatic Differentiation:** Implements a topological sort to accurately apply the chain rule backwards through the computational graph[cite: 1].
* **Supported Operations:** Addition, multiplication, matrix multiplication, exponential, logarithm, sigmoid, and hyperbolic tangent (tanh)[cite: 1].
* **Neural Network Layers:**
  * `Linear`: Fully connected dense layers[cite: 1].
  * `Conv2D`: Convolutional layers featuring custom sliding-window implementations[cite: 1].
  * `ReLU`: Activation functions[cite: 1].
  * `Flatten`: For transitioning between convolutional and linear layers[cite: 1].
* **Optimizers:** `SGD` (Stochastic Gradient Descent) and `Adam`[cite: 1].
* **Loss Function:** Cross-Entropy Loss with integrated Softmax[cite: 1].

## Experiments
The engine is validated against two standard datasets:
1. **Two Moons:** Training a Multi-Layer Perceptron (MLP) on a non-linear 2D dataset[cite: 1].
2. **MNIST:** Training both an MLP and a Convolutional Neural Network (CNN) for handwritten digit classification, achieving high accuracy[cite: 1].

## Usage
Open `autograd_using_numpy.ipynb` in Jupyter Notebook or JupyterLab to execute the cells and view the training loops and outputs.
