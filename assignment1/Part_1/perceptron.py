import numpy as np
import matplotlib.pyplot as plt


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

    def forward(self, input_vec) -> int:
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted labels.
        """
        return np.sign(np.dot(input_vec, self.weights) + self.bias)

    def train(self, training_inputs, labels, test_inputs=None, test_labels=None) -> list:
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
            test_inputs:
            test_labels:
        """
        n = len(training_inputs)
        accuracy = [[], []]
        for epoch in range(self.max_epochs):
            """
                in one epoch:
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            # train
            count = 0
            grad = np.zeros(self.weights.shape)
            grad_bias = 0
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
            accuracy[0].append((n - count) / n)

            # test
            if test_inputs is not None and test_labels is not None:
                accuracy[1].append(self.test(test_inputs, test_labels))

            # output
            if epoch % 100 == 0:
                print("epoch", epoch, "- wrong:", count)

        return accuracy

    def test(self, test_inputs, test_labels) -> float:
        n = len(test_inputs)
        count = 0
        for xi, yi in zip(test_inputs, test_labels):
            prediction = self.forward(xi)
            count += (prediction != yi)
        return (n - count) / n

    def get_k_b(self):
        """
        x * w1 + y * w2 + b = 0, so we can have:
        y = - (w1 / w2) x - (b / w2)
        k = - (w1 / w2)
        b = - (b / w2)

        Returns: k and b values corresponding
        """
        k = - self.weights[0] / self.weights[1]
        b = - self.bias / self.weights[1]
        return k, b


if __name__ == "__main__":
    # dataset
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

    # perceptron train and test
    perceptron = Perceptron(np.shape(dataset_train[0][:-1]))
    acc = perceptron.train(list(dataset_train[:, :-1]), dataset_train[:, -1:],
                           list(dataset_train[:, :-1]), dataset_train[:, -1:])
    acc_train, acc_test = acc[0], acc[1]

    # plot 1: accuracy curve [Using ChatGPT]
    x_values = [i for i in range(len(acc_train))]
    plt.plot(x_values, acc_train, label='train accuracy')
    plt.plot(x_values, acc_test, label='test accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    # plot 2: points map with perceptron line [Using ChatGPT]
    data = dataset
    x_values = [point[0] for point in data]
    y_values = [point[1] for point in data]
    labels = [point[2] for point in data]
    x_values_label1 = [x_values[i] for i in range(len(data)) if labels[i] == 1]
    y_values_label1 = [y_values[i] for i in range(len(data)) if labels[i] == 1]
    x_values_label_minus1 = [x_values[i] for i in range(len(data)) if labels[i] == -1]
    y_values_label_minus1 = [y_values[i] for i in range(len(data)) if labels[i] == -1]
    plt.scatter(x_values_label1, y_values_label1, color='blue', label='Label 1')
    plt.scatter(x_values_label_minus1, y_values_label_minus1, color='red', label='Label -1')
    k, b = perceptron.get_k_b()
    x_line = range(int(min(x_values) - 1), int(max(x_values) + 1))
    y_line = [k * x + b for x in x_line]
    plt.plot(x_line, y_line, color='green', label='Line')
    plt.title('Points Map with Perceptron Line')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()
