#!/usr/bin/env python3


from keras.datasets import mnist

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_train = (X_train - 127.5) / 127.5 # Normalize the images to [-1, 1]

    return X_train, y_train, X_test, y_test
