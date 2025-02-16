import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self):
        self.activation_function = lambda x: 1.5*math.tanh(x) + 1
        self.weights = []
        self.bias = 0

    def forward(self, x: list[float]) -> float:
        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def train(self, X_train: list[list[float]], y_expected: list[float], n: int, learning_rate: float):

        number_of_inputs = len(X_train[0])
        self.weights = np.random.randn(number_of_inputs)
        self.bias = np.random.randn()
        errors = []

        for epoch in range(n):
            total_error = 0
            for i, x in enumerate(X_train):
                y_predicted = self.forward(x)
                error = y_expected[i] - y_predicted
                correction = error * learning_rate

                self.weights += correction * x
                self.bias += correction
                total_error += error**2  

            errors.append(total_error / len(X_train))

            if epoch % (n // 10) == 0: 
                print(f"Epoch {epoch}, Error: {errors[-1]}")

        

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self.forward(x) for x in X]