import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math

class Sampler(object):
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(logits, temperature):
        y = logits + Sampler.sample_gumbel(logits.size()).to(logits.device)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(logits, temperature):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = Sampler.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y


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
        self.encoder = nn.Embedding(ntoken, ninp)
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
        self.model = TransformerModel(631, n_words, 2, 120, 6)

    def forward(self, x):
        in_shape = x.shape
        x = self.model(x)
        x = Sampler.gumbel_softmax(x, 0.1)
        x = torch.nonzero(x, as_tuple=True)[2].view(in_shape)
        return x

class Discriminator(nn.Module):
    def __init__(self, n_words):
        super(Discriminator, self).__init__()


        self.custom_transformer = True
        self.model = TransformerModel(631, n_words, 2, 120, 6)
        self.linear_out = nn.Linear(631 * n_words, 1)

    def forward(self, x):
        x = self.model(x)
        x = x.transpose(0, 1).contiguous().view(x.shape[1], -1)
        x = self.linear_out(x)
        x = torch.sigmoid(x).flatten()
        return x

class CycleLoss(nn.Module):

    def __init__(self, n_words):
        super(CycleLoss, self).__init__()

        self.model = TransformerModel(631, n_words*2, 2, 160, 6)
        self.linear_out = nn.Linear(631*n_words*2, 1)

    def forward(self, x, y):
        x = torch.cat([x, y])
        x = self.model(x)
        x = x.transpose(0, 1).contiguous().view(x.shape[1], -1)
        x = self.linear_out(x)
        x = torch.sigmoid(x).view(-1)
        return x
