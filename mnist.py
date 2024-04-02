"""This file contains the code to train a neural network on the mnist dataset 
using the CustomNeuralNetwork class."""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import cupy as np
from NN import NeuralNetwork
from layers import DenseLayer, FlattenLayer, CnnLayer, MaxPoolingLayer
from lossFuncs import sparse_categorical_crossentropy
from activationFuncs import relu, softmax


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    X = X.reshape(-1, 8, 8, 1)
    X = X / 255
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    X_train = X_train[:10]
    y_train = y_train[:10]
    model = NeuralNetwork((8, 8, 1), lr=0.01, loss=sparse_categorical_crossentropy)
    model.add_layer(CnnLayer,
                    kernel_size=(4, 4),
                    filters=2,
                    stride=1,
                    padding=1,
                    activation=relu,
                    input_channels=1)
    model.add_layer(MaxPoolingLayer, pool_size=2, stride=2)
    model.add_layer(FlattenLayer)
    model.add_layer(DenseLayer, output_size=10, activation=softmax)
    model.train(X_train, y_train, epochs=150)
    preds = model(X_train)
    print(preds)
    print(y_train)
