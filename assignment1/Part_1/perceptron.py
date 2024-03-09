import numpy as np


class Perceptron(object):
    def __init__(self, n_inputs, max_epochs=1e3, learning_rate=1e-2):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs = n_inputs  # Fill in: Initialize number of inputs
        self.max_epochs = int(max_epochs)  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        self.weights = np.zeros(n_inputs)  # Fill in: Initialize weights with zeros

    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted labels.
        """
        return np.sign(np.dot(input_vec, self.weights))

    def train(self, training_inputs, labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        for _ in range(self.max_epochs):
            """
                in one epoch:
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            tmp = np.column_stack((training_inputs, labels))
            np.random.shuffle(tmp)
            training_inputs = tmp[:, :-1]
            labels = tmp[:, -1]
            pred = self.forward(training_inputs)
            if np.dot(labels, pred) <= 0:
                derivation = np.dot(labels, training_inputs)
                self.weights += self.learning_rate * derivation
