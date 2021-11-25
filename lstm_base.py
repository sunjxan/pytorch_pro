import torch
import torch.nn as nn

import math

from collections import OrderedDict


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
    # hcx: (hx, cx)
    # hx, cx, h, c: (batch, hidden_size)
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
        g = torch.sigmoid(g)
        o = torch.sigmoid(o)

        c = i * g
        if cx is not None:
            c += f * cx
        h = o * torch.tanh(c)
        return h, c

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
    # hcx: (hx, cx)
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
    # hcxs: (num_layers, batch, 2, hidden_size)
    # outputs: (seq, batch, hidden_size)
    # output_hcxs: (num_layers, batch, 2, hidden_size)
    def forward(self, inputs, hcxs=None):
        if hcxs is None:
            hcxs = [None] * self.num_layers
        output_hxs = []
        output_cxs = []
        for i in range(self.num_layers):
            inputs, (hx, cx) = self.layers[i](inputs, hcxs[i])
            output_hxs.append(hx)
            output_cxs.append(cx)
        return inputs, (torch.stack(output_hxs), torch.stack(output_cxs))


rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output.shape, hn.shape, cn.shape)

rnn = LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output.shape, hn.shape, cn.shape)

# input = torch.randn(5, 2, 3)
# h0 = torch.randn(2, 2, 4)
# c0 = torch.rand(2, 2, 4)

# # 验证
# LSTM = LSTM(3, 4, 2)
# L = [list(cell.parameters()) for cell in LSTM.layers]
# size = len(L)
# D = []
# for i in range(size):
#     D.append(['weight_ih_l' + str(i), L[i][0]])
#     D.append(['weight_hh_l' + str(i), L[i][2]])
#     D.append(['bias_ih_l' + str(i), L[i][1]])
#     D.append(['bias_hh_l' + str(i), L[i][3]])
# D = OrderedDict(D)

# # PyTorch LSTM
# LSTM2 = nn.LSTM(3, 4, 2)
# LSTM2.load_state_dict(D)

# output, hn = LSTM(input, h0)
# output2, hn2 = LSTM2(input, h0)
# print(torch.all(output == output2), torch.all(hn == hn2))
