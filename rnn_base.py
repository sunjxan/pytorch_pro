import torch
import torch.nn as nn

import math

from collections import OrderedDict


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, tanh=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.tanh = tanh
        self.fc_ih = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hh = nn.Linear(hidden_size, hidden_size, bias=bias)
        self._initialize_weights()

    # input: (batch, input_size)
    # hx: (batch, hidden_size)
    # output: (batch, hidden_size)
    def forward(self, input, hx=None):
        output = self.fc_ih(input)
        if hx is not None:
            output += self.fc_hh(hx)
        output = torch.tanh(output) if self.tanh else torch.relu(output)
        return output

    def _initialize_weights(self):
        k = math.sqrt(1 / self.hidden_size)
        nn.init.uniform_(self.fc_ih.weight, -k, k)
        nn.init.uniform_(self.fc_hh.weight, -k, k)
        if self.bias:
            nn.init.uniform_(self.fc_ih.bias, -k, k)
            nn.init.uniform_(self.fc_hh.bias, -k, k)

class RNNSection(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, tanh=True):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size, bias, tanh)

    # inputs: (seq, batch, input_size)
    # hx: (batch, hidden_size)
    # outputs: (seq, batch, hidden_size)
    def forward(self, inputs, hx=None):
        outputs = []
        for input in inputs:
            hx = self.cell(input, hx)
            outputs.append(hx)
        return torch.stack(outputs), hx

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, tanh=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(RNNSection(input_size, hidden_size, bias, tanh))
            else:
                self.layers.append(RNNSection(hidden_size, hidden_size, bias, tanh))

    # inputs: (seq, batch, input_size)
    # hxs: (num_layers, batch, hidden_size)
    # outputs: (seq, batch, hidden_size)
    # output_hxs: (num_layers, batch, hidden_size)
    def forward(self, inputs, hxs=None):
        output_hxs = []
        for i in range(self.num_layers):
            inputs, output_hx = self.layers[i](inputs, None if hxs is None else hxs[i])
            output_hxs.append(output_hx)
        return inputs, torch.stack(output_hxs)


input = torch.randn(5, 2, 3)
h0 = torch.randn(2, 2, 4)

# 验证
rnn = RNN(3, 4, 2)
L = [list(cell.parameters()) for cell in rnn.layers]
size = len(L)
D = []
for i in range(size):
    D.append(['weight_ih_l' + str(i), L[i][0]])
    D.append(['weight_hh_l' + str(i), L[i][2]])
    D.append(['bias_ih_l' + str(i), L[i][1]])
    D.append(['bias_hh_l' + str(i), L[i][3]])
D = OrderedDict(D)

# PyTorch RNN
rnn2 = nn.RNN(3, 4, 2)
rnn2.load_state_dict(D)

output, hn = rnn(input, h0)
output2, hn2 = rnn2(input, h0)
print(torch.all(output == output2), torch.all(hn == hn2))
