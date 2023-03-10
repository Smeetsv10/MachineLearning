import numpy as np
import matplotlib.pyplot as plt

import numpy as np

# Define the neural network class


class NeuralNetwork():
    def __init__(self):
        # Initialize the weights randomly
        self.weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        # Activation function: sigmoid
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        return x * (1 - x)

    def train(self, inputs, outputs, num_iterations):
        for i in range(num_iterations):
            # Forward pass
            output = self.predict(inputs)

            loss = np.mean(np.square(output - outputs))

            # Backward pass
            error = outputs - output
            adjustment = error * self.sigmoid_derivative(output)
            self.weights += np.dot(inputs.T, adjustment)

            if i % 100 == 0:
                print(f"Epoch: {i}, Loss: {loss:.4f}")

    def predict(self, inputs):
        # Pass inputs through the neural network
        return self.sigmoid(np.dot(inputs, self.weights))


# Example usage
if __name__ == '__main__':
    # Define training data
    t = np.load('time_train.npy')
    x = np.load('input_train.npy')
    x = x.T
    y = np.load('output_train.npy')
    y = y / np.mean(y)
    # Create a neural network instance and train it
    neural_network = NeuralNetwork()
    neural_network.train(x, y, 10000)

    # Predict the output for new input
    t = np.load('time_validate.npy')
    x = np.load('input_validate.npy')
    x = x.T
    y = np.load('output_validate.npy')
    y = y / np.mean(y)
    
    predicted_output = neural_network.predict(x)

    plt.plot(t, predicted_output, label='Predicted')
    plt.plot(t, y, label='True value')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.show()
