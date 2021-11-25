import torch
import torch.nn as nn

import math


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.fc_ir = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hr = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_iz = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hz = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hn = nn.Linear(hidden_size, hidden_size, bias=bias)

        self._initialize_weights()

    # input: (batch, input_size)
    # hx, hn: (batch, hidden_size)
    def forward(self, input, hx=None):
        r = self.fc_ir(input)
        z = self.fc_iz(input)
        n = self.fc_in(input)
        if hx is not None:
            r += self.fc_hr(hx)
            z += self.fc_hz(hx)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        if hx is not None:
            n += r * self.fc_hn(hx)
        n = torch.tanh(n)

        hn = (1 - z) * n
        if hx is not None:
            hn += z * hx
        return hn

    def _initialize_weights(self):
        k = math.sqrt(1 / self.hidden_size)
        nn.init.uniform_(self.fc_ir.weight, -k, k)
        nn.init.uniform_(self.fc_hr.weight, -k, k)
        nn.init.uniform_(self.fc_iz.weight, -k, k)
        nn.init.uniform_(self.fc_hz.weight, -k, k)
        nn.init.uniform_(self.fc_in.weight, -k, k)
        nn.init.uniform_(self.fc_hn.weight, -k, k)
        if self.bias:
            nn.init.uniform_(self.fc_ir.bias, -k, k)
            nn.init.uniform_(self.fc_hr.bias, -k, k)
            nn.init.uniform_(self.fc_iz.bias, -k, k)
            nn.init.uniform_(self.fc_hz.bias, -k, k)
            nn.init.uniform_(self.fc_in.bias, -k, k)
            nn.init.uniform_(self.fc_hn.bias, -k, k)

class GRUSection(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.cell = GRUCell(input_size, hidden_size, bias)

    # inputs: (seq, batch, input_size)
    # hx: (batch, hidden_size)
    # outputs: (seq, batch, hidden_size)
    def forward(self, inputs, hx=None):
        outputs = []
        for input in inputs:
            hx = self.cell(input, hx)
            outputs.append(hx)
        return torch.stack(outputs), hx

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GRUSection(input_size, hidden_size, bias))
            else:
                self.layers.append(GRUSection(hidden_size, hidden_size, bias))

    # inputs: (seq, batch, input_size)
    # hxs, hns: (num_layers, batch, hidden_size)
    # outputs: (seq, batch, hidden_size)
    def forward(self, inputs, hxs=None):
        hns = []
        for i in range(self.num_layers):
            inputs, hn = self.layers[i](inputs, hxs and hxs[i])
            hns.append(hn)
        return inputs, torch.stack(hns)
