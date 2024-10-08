import time

import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features, learning_rate=1e-2):
        """
        Initializes a linear (fully connected) layer. 
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        Args:
            in_features: number of input
            out_features: number of output
        """
        self.x = None
        self.in_features = in_features
        self.out_features = out_features
        self.lr = learning_rate
        self.params = {
            'weight': np.random.randn(in_features, out_features) * 1e-2,
            'bias': np.random.randn(1, out_features) * 1e-2
        }
        self.grads = {
            'weight': np.zeros((in_features, out_features)),
            'bias': np.zeros((1, out_features))
        }

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        Args:
            x: input data
        """
        self.x = x
        return np.dot(x, self.params['weight']) + self.params['bias']

    def backward(self, dout):
        """
        Backward propagation to calculate gradients of loss w.r.t. weights and inputs.
        Args:
            dout: Upstream derivative
        """
        self.grads['weight'] += np.dot(self.x.T, dout)  # dw
        self.grads['bias'] += np.sum(dout, axis=0)  # db
        dx = np.dot(dout, self.params['weight'].T)
        return dx

    def update(self):
        size = 1  # len(self.x)
        self.params['weight'] -= self.lr * self.grads['weight'] / size
        self.params['bias'] -= self.lr * self.grads['bias'] / size
        self.grads['weight'] = np.zeros_like(self.grads['weight'])
        self.grads['bias'] = np.zeros_like(self.grads['bias'])

    def __call__(self, x):
        return self.forward(x)


class ReLU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        Args:
            x: input data
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        Args:
            dout: Upstream derivative
        """
        return np.where(self.x > 0, dout, 0)

    def __call__(self, x):
        return self.forward(x)


class SoftMax(object):
    def forward(self, x: np.ndarray):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        Args:
            x: input data
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def backward(self, dout):
        """
        The backward propagation for softmax is often directly integrated with CrossEntropy for simplicity.
        Args:
            dout: Upstream derivative
        """
        return dout

    def __call__(self, x):
        return self.forward(x)


class CrossEntropy(object):
    def forward(self, x: np.ndarray, y: np.ndarray):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        Args:
            x: input data
            y: true labels
        """
        return -np.sum(y * np.log(x))

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        return x - y

    def __call__(self, x, y):
        return self.forward(x, y)
