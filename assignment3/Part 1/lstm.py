from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.input_length = input_length
        self.hidden_dim = hidden_dim

        # Input gate
        self.W_xi = nn.Linear(input_dim, hidden_dim, True)
        self.W_hi = nn.Linear(hidden_dim, hidden_dim, True)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))
        # Forget gate
        self.W_xf = nn.Linear(input_dim, hidden_dim, True)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim, True)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))
        # Cell gate
        self.W_xc = nn.Linear(input_dim, hidden_dim, True)
        self.W_hc = nn.Linear(hidden_dim, hidden_dim, True)
        self.b_c = nn.Parameter(torch.zeros(hidden_dim))
        # Output gate
        self.W_xo = nn.Linear(input_dim, hidden_dim, True)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim, True)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
        # Output layer
        self.W_hp = nn.Linear(hidden_dim, output_dim, True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(batch_size, self.hidden_dim)
        cell = torch.zeros(batch_size, self.hidden_dim)
        for i in range(self.input_length):
            x_t = x[:, i]
            # Input gate
            i_t = torch.sigmoid(self.W_xi(x_t) + self.W_hi(hidden) + self.b_i)
            # Forget gate
            f_t = torch.sigmoid(self.W_xf(x_t) + self.W_hf(hidden) + self.b_f)
            # Cell gate
            g_t = torch.tanh(self.W_xc(x_t) + self.W_hc(hidden) + self.b_c)
            # Output gate
            o_t = torch.sigmoid(self.W_xo(x_t) + self.W_ho(hidden) + self.b_o)
            # Update cell state
            cell = f_t * cell + i_t * g_t
            # Update hidden state
            hidden = o_t * torch.tanh(cell)

        # Output layer
        output = self.W_hp(hidden)
        return self.softmax(output)
