import torch
import torch.nn as nn

import math


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.fc_ii = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hi = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_if = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hf = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_ig = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hg = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_io = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_ho = nn.Linear(hidden_size, hidden_size, bias=bias)

        self._initialize_weights()

    # input: (batch, input_size)
    # hcx = (hx, cx)
    # hx, cx, hn, cn: (batch, hidden_size)
    def forward(self, input, hcx=None):
        hx = cx = None
        if hcx is not None:
            hx, cx = hcx
        
        i = self.fc_ii(input)
        f = self.fc_if(input)
        g = self.fc_ig(input)
        o = self.fc_io(input)
        if hx is not None:
            i += self.fc_hi(hx)
            f += self.fc_hf(hx)
            g += self.fc_hg(hx)
            o += self.fc_ho(hx)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        cn = i * g
        if cx is not None:
            cn += f * cx
        hn = o * torch.tanh(cn)
        return hn, cn

    def _initialize_weights(self):
        k = math.sqrt(1 / self.hidden_size)
        nn.init.uniform_(self.fc_ii.weight, -k, k)
        nn.init.uniform_(self.fc_hi.weight, -k, k)
        nn.init.uniform_(self.fc_if.weight, -k, k)
        nn.init.uniform_(self.fc_hf.weight, -k, k)
        nn.init.uniform_(self.fc_ig.weight, -k, k)
        nn.init.uniform_(self.fc_hg.weight, -k, k)
        nn.init.uniform_(self.fc_io.weight, -k, k)
        nn.init.uniform_(self.fc_ho.weight, -k, k)
        if self.bias:
            nn.init.uniform_(self.fc_ii.bias, -k, k)
            nn.init.uniform_(self.fc_hi.bias, -k, k)
            nn.init.uniform_(self.fc_if.bias, -k, k)
            nn.init.uniform_(self.fc_hf.bias, -k, k)
            nn.init.uniform_(self.fc_ig.bias, -k, k)
            nn.init.uniform_(self.fc_hg.bias, -k, k)
            nn.init.uniform_(self.fc_io.bias, -k, k)
            nn.init.uniform_(self.fc_ho.bias, -k, k)

class LSTMSection(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size, bias)

    # inputs: (seq, batch, input_size)
    # hcx = (hx, cx)
    # hx, cx: (batch, hidden_size)
    # outputs: (seq, batch, hidden_size)
    def forward(self, inputs, hcx=None):
        hx = cx = None
        if hcx is not None:
            hx, cx = hcx
        outputs = []
        for input in inputs:
            hx, cx = self.cell(input, (hx, cx))
            outputs.append(hx)
        return torch.stack(outputs), (hx, cx)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(LSTMSection(input_size, hidden_size, bias))
            else:
                self.layers.append(LSTMSection(hidden_size, hidden_size, bias))

    # inputs: (seq, batch, input_size)
    # hcxs = (hxs, cxs)
    # hxs, cxs, hns, cns: (num_layers, batch, hidden_size)
    # outputs: (seq, batch, hidden_size)
    def forward(self, inputs, hcxs=None):
        if hcxs is not None:
            hxs, cxs = hcxs
        hns = []
        cns = []
        for i in range(self.num_layers):
            inputs, (hn, cn) = self.layers[i](inputs, None if hcxs is None else (hxs[i], cxs[i]))
            hns.append(hn)
            cns.append(cn)
        return inputs, (torch.stack(hns), torch.stack(cns))
