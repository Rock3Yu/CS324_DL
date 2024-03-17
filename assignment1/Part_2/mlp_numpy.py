from typing import List
import numpy as np

from modules import Linear, ReLU, SoftMax, CrossEntropy


class MLP(object):
    def __init__(self, n_inputs: int, n_hidden: List[int], n_classes: int, learning_rate=1e-2):
        """
        Initializes the multi-layer perceptron object.

        This function should initialize the layers of the MLP including any linear layers and activation functions
        you plan to use. You will need to create a list of linear layers based on n_inputs, n_hidden, and n_classes.
        Also, initialize ReLU activation layers for each hidden layer and a softmax layer for the output.

        Args:
            n_inputs: Number of inputs (i.e., dimension of an input vector).
            n_hidden: List of integers, where each integer is the number of units in each hidden layer.
            n_classes: Number of classes of the classification problem (i.e., output dimension of the network).
            learning_rate: learning rate
        """
        input_layer = [Linear(n_inputs, n_hidden[0], learning_rate), ReLU()]
        hidden_layers = []
        output_layer = [Linear(n_hidden[-1], n_classes, learning_rate), SoftMax()]
        for i in range(len(n_hidden) - 1):
            hidden_layers += [Linear(n_hidden[i], n_hidden[i + 1], learning_rate), ReLU()]

        self.layers = input_layer + hidden_layers + output_layer
        self.loss_fn = CrossEntropy()

    def forward(self, x: np.ndarray, predict=False) -> np.ndarray:
        """
        Predicts the network output from the input by passing it through several layers.
        
        Here, you should implement the forward pass through all layers of the MLP. This involves
        iterating over your list of layers and passing the input through each one sequentially.
        Don't forget to apply the activation function after each linear layer except for the output layer.
        
        Args:
            x: Input to the network.
            predict: just predict or forward (may update params)
        Returns:
            numpy.ndarray: Output of the network.
        """
        out = x
        for layer in self.layers:
            out = layer(out, predict=predict)  # __call__() will invoke def forward()
        return out

    def backward(self, dout: np.ndarray) -> None:
        """
        Performs the backward propagation pass given the loss gradients.
        
        Here, you should implement the backward pass through all layers of the MLP. This involves
        iterating over your list of layers in reverse and passing the gradient through each one sequentially.
        You will update the gradients for each layer.
        
        Args:
            dout (numpy.ndarray): Gradients of the loss with respect to the output of the network.
        """
        # No need to return anything since the gradients are stored in the layers.
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def __call__(self, x: np.ndarray, predict=False) -> np.ndarray:
        return self.forward(x, predict)
