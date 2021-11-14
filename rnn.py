import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import random, time


class RNN:
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x, hidden):
        return self.step(x, hidden)

    def step(self, x, hidden):
        h1 = self.W_hh(hidden)
        w1 = self.W_xh(x)
        out = torch.tanh(h1 + w1)
        hidden = self.W_hh.weight
        return out, hidden

rnn = RNN(20, 50)
input = torch.randn(32, 20)
h_0 = torch.randn(32, 50)
seq_len = input.shape[0]
for i in range(seq_len):
    output, hn = rnn(input[i, :], h_0)
print(output.size(), h_0.size())
