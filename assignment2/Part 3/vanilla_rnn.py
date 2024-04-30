from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """
    (1) h_t = tanh(W_hx * x_t + W_hh * h_{t-1} + b_h)
    (2) o_t = W_ph * h_t + b_o
    (3) y_t^{~} = softmax(o_t)
    """

    def __init__(self, input_length, input_dim, hidden_dim, output_dim, device="cpu", batch_size=10):
        super(VanillaRNN, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # para of (1)
        self.W_hx = nn.Parameter(torch.zeros((hidden_dim, input_dim)).to(device), True)
        self.W_hh = nn.Parameter(torch.zeros((hidden_dim, hidden_dim)).to(device), True)
        self.b_h = nn.Parameter(torch.zeros((hidden_dim, 1)).to(device), True)
        # para of (2)
        self.W_ph = nn.Parameter(torch.zeros((output_dim, hidden_dim)).to(device), True)
        self.b_o = nn.Parameter(torch.zeros((output_dim, 1)).to(device), True)
        # h_i, where i is time t, initial was h_0 = zeros
        self.h_i = nn.Parameter(torch.zeros(hidden_dim, batch_size).to(device), True)
        # partial weights initialization
        nn.init.xavier_uniform_(self.W_hx)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.W_ph)

    def forward(self, x):
        nn.init.zeros_(self.h_i)
        h_t = self.h_i
        o_t = None
        for t in range(self.input_length):
            x_t = x[:, t].view(1, -1)  # batch
            h_t = torch.tanh(torch.matmul(self.W_hx, x_t) +
                             torch.matmul(self.W_hh, h_t) +
                             self.b_h)
            o_t = torch.matmul(self.W_ph, h_t) + self.b_o
        y = torch.softmax(o_t.T, dim=-1)
        return y
