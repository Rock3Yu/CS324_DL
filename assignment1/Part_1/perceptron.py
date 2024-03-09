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
        self.bias = 0

    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted labels.
        """
        return np.sign(np.dot(input_vec, self.weights) + self.bias)

    def train(self, training_inputs, labels, use_shuffle=False):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
            use_shuffle: Whether to shuffle the training_data
        """
        n = len(training_inputs)
        for epoch in range(self.max_epochs):
            """
                in one epoch:
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            count = 0
            grad = np.zeros(self.weights.shape)
            grad_bias = 0
            if use_shuffle:
                state = np.random.get_state()
                np.random.shuffle(training_inputs)
                np.random.set_state(state)
                np.random.shuffle(labels)
            for xi, yi in zip(training_inputs, labels):
                prediction = self.forward(xi)
                if prediction != yi:
                    grad -= xi * yi
                    grad_bias -= yi
                    count += 1
            grad /= n
            grad_bias /= n
            self.weights -= self.learning_rate * grad
            self.bias -= self.learning_rate * grad_bias
            if epoch % 100 == 0:
                print("epoch", epoch, "- wrong:", count)


if __name__ == "__main__":
    dataset = []
    centers = [[30, 27], [10, 7]]
    stds = [[1, 10], [3, 3]]

    for i in range(100):
        x0 = np.random.normal(centers[0][0], stds[0][0])
        y0 = np.random.normal(centers[0][1], stds[0][1])
        x1 = np.random.normal(centers[1][0], stds[1][0])
        y1 = np.random.normal(centers[1][1], stds[1][1])
        dataset.append([x0, y0, -1])
        dataset.append([x1, y1, +1])

    dataset_train = np.array(dataset[:160])
    dataset_test = np.array(dataset[160:])

    perceptron = Perceptron(np.shape(dataset_train[0][:-1]))
    perceptron.train(list(dataset_train[:, :-1]), dataset_train[:, -1:])
