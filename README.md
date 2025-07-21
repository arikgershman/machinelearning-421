# Neural Networks for Machine Learning 🤖🧠

This repository contains my solutions for Project 5: Machine Learning, a deep learning assignment from my CMSC 421: Introduction to Artificial Intelligence course at the University of Maryland. In this project, I built neural networks from scratch to tackle various machine learning tasks, demonstrating an understanding of fundamental concepts in deep learning.

## Project Overview

The project involved implementing core neural network components and applying them to practical problems. My work focused on building models for classification and regression tasks, including a Perceptron, a non-linear regression model, a digit classifier, and a language identification model using a simple Recurrent Neural Network (RNN).

I successfully achieved a perfect score of **26/26** on the project's autograder tests.

## Files Included

The key files in this repository are:

* `models.py`: This file contains my implementations of the Perceptron and various neural network models. **(This is where the core solutions reside)**
* `nn.py`: This file contains a basic neural network mini-library with definitions for various node objects (e.g., `Constant`, `Parameter`, `Add`, `Linear`, `ReLU`, `SquareLoss`, `SoftmaxLoss`) and functions for computing gradients. **(This file was provided, but is crucial for understanding the underlying framework.)**
* `backend.py`: Backend code for various machine learning tasks.
* `data/`: Contains datasets necessary for digit classification and language identification. **(Included as it's essential for running the models.)**

## Getting Started

To run the project:

1.  **Download:** Clone this repository to your local machine.
2.  **Dependencies:** Ensure you have `numpy` and `matplotlib` installed. If using `conda`, you can install them via:
    ```bash
    conda activate [your_environment_name]
    pip install numpy matplotlib
    ```
3.  **Run Models:** The models are evaluated using an autograder (provided in the original course distribution). While the autograder itself is not included in this public repository for academic integrity reasons, the models can be run and tested within the original project environment.

    *(Optional: If you have the full course distribution, you can run `python autograder.py` to verify solutions.)*

## Implemented Models & Key Concepts

I implemented the following models and demonstrated proficiency in these neural network concepts:

### 1. Perceptron

* **Task:** Binary classification.
* **Implementation:** Completed the `PerceptronModel` class in `models.py`, including `run()`, `get_prediction()`, and `train()` methods. Utilized `nn.DotProduct` for computing scores and `nn.Parameter.update()` for weight adjustments.

### 2. Non-linear Regression

* **Task:** Approximate `sin(x)` over `[-2π, 2π]`.
* **Architecture:** Employed a neural network (typically a shallow network with one hidden layer and two linear layers) and `nn.SquareLoss` as the loss function.
* **Implementation:** Completed the `RegressionModel` class in `models.py`, defining the network architecture, forward pass (`run`), loss calculation (`get_loss`), and training loop (`train`) using gradient-based updates.

### 3. Digit Classification

* **Task:** Classify handwritten digits from the MNIST dataset.
* **Input/Output:** Handled 784-dimensional input vectors (28x28 pixels) and produced 10-dimensional output scores for 0-9 classes.
* **Architecture:** Used `nn.SoftmaxLoss` and avoided ReLU in the final layer for score output. My solution achieved an accuracy of at least 97% on the test set.

### 4. Language Identification (Recurrent Neural Network - RNN)

* **Task:** Identify the language of single words (e.g., English, Spanish, Finnish, Dutch, Polish).
* **Key Challenge:** Handling variable-length inputs (words with different numbers of letters).
* **Approach:** Implemented a simple Recurrent Neural Network (RNN) structure using a shared neural network function `f` that processes characters sequentially and updates a hidden state, summarizing the word. The final hidden state is then used for classification.
* **Implementation:** Completed the `LanguageIDModel` class, demonstrating an understanding of recurrent neural network principles and batch processing for variable-length sequences.

## Neural Network Library (`nn.py`)

The `nn.py` file defines the foundational components used across all models:

* **`Node` Hierarchy:** `DataNode` (parent for `Parameter`, `Constant`) and `FunctionNode` (parent for operations like `Add`, `Linear`, `ReLU`, `Loss` functions).
* **Key Operations:**
    * `nn.Parameter(n, m)`: Creates trainable weight matrices.
    * `nn.Add(x, y)`: Element-wise matrix addition.
    * `nn.AddBias(features, bias)`: Adds a bias vector to feature rows.
    * `nn.Linear(features, weights)`: Performs matrix multiplication.
    * `nn.ReLU(features)`: Applies the Rectified Linear Unit non-linearity.
    * `nn.SquareLoss(a, b)`: Computes mean squared error (for regression).
    * `nn.SoftmaxLoss(logits, labels)`: Computes softmax cross-entropy loss (for classification).
* **`nn.gradients(loss, parameters)`:** My implementation leverages this core function for backpropagation, computing gradients of the loss with respect to network parameters.
* **`nn.as_scalar(node)`:** Utility to extract a Python float from a scalar Node.

This project provided a robust introduction to building and training neural networks using fundamental operations, which is a great foundation for further exploration in machine learning.
