from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from torch.nn.modules import Linear, ReLU, Softmax, CrossEntropyLoss, Sequential


class MLP(nn.Module):

    def __init__(self, n_inputs: int, n_hidden: list, n_classes: int):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        input_layer = [Linear(n_inputs, n_hidden[0]), ReLU()]
        hidden_layer = []
        output_layer = [Linear(n_hidden[-1], n_classes), Softmax()]
        for i in range(len(n_hidden) - 1):
            hidden_layer += [Linear(n_hidden[i], n_hidden[i + 1]), ReLU()]

        self.layers = input_layer + hidden_layer + output_layer
        self.layers = Sequential(*self.layers)  # * to deconstruct the list
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        return self.layers(x)
