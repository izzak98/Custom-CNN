# Neural Network Library for MNIST Classification

This project implements a custom neural network library in Python, utilizing CuPy for GPU-accelerated computations. Designed specifically for educational purposes, this library offers a hands-on experience with the inner workings of neural networks. It includes various components like layers, activation functions, loss functions, and optimizers, with a focus on classifying the MNIST dataset.

## Features

- GPU acceleration with CuPy
- Customizable neural network architecture
- Support for dense and convolutional layers
- Implementation of ReLU, sigmoid, tanh, and softmax activation functions
- Mean squared error and sparse categorical crossentropy loss functions
- Adam optimizer
- MNIST classification example

## Requirements

- CuPy
- tqdm
- scikit-learn

## Installation

Ensure that you have Python 3.9 or greater installed on your system. You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

Note to utilize GPU acceleration please refer to the [official cupy installation guide](https://docs.cupy.dev/en/stable/install.html).
## Usage

To train a neural network on the MNIST dataset, you can use the `mnist.py` script as follows:

```bash
python mnist.py
```

This script initializes a neural network with a specific architecture tailored for MNIST digit classification, trains the model on the dataset, and prints the predictions for a subset of the training data.

You can also create custom neural network architectures by importing and utilizing the classes provided in `NN.py`, `layers.py`, `activationFuncs.py`, `lossFuncs.py`, and `Optmizers.py`.
