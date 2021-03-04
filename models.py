import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


class Generator(nn.Module):
    def __init__(self, n_words):
        super(Generator, self).__init__()

        self.custom_transformer = True
        self.model = TransformerModel(631, n_words, 2, 120, 2)
    def forward(self, x, temp=0.1):
        in_shape = x.shape
        x = self.model(x)
        x = F.log_softmax(x, dim=-1)
        x = F.gumbel_softmax(x, tau=temp, hard=True)

        return x

class Discriminator(nn.Module):
    def __init__(self, n_words):
        super(Discriminator, self).__init__()

        self.n_words = n_words
        self.model = TransformerModel(631, n_words, 2, 120, 2)
        self.linear_out = nn.Linear(631 * 40, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.transpose(0, 1).contiguous().view(x.shape[1], -1)
        x = self.linear_out(x)
        x = F.softmax(x, dim=-1)
        return x

class CycleLoss(nn.Module):

    def __init__(self, n_words):
        super(CycleLoss, self).__init__()

        self.model = TransformerModel(631, n_words*2, 2, 120, 2)
        self.linear_out = nn.Linear(631*40*2, 2)

    def forward(self, x, y):
        x = torch.cat([x, y])
        x = self.model(x)
        x = x.transpose(0, 1).contiguous().view(x.shape[1], -1)
        x = self.linear_out(x)
        x = F.softmax(x, dim=-1)
        return x
