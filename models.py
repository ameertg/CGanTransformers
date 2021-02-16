import torch.nn as nn
import torch.load as load
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

class Generator(nn.Module):
    def __init__(self, transformer):
        super(Generator, self).__init__()

        # Load transformer
        transformer = torch.load(transformer, torch.device('cpu'))
        self.model = transformer

    def forward(self, x):
        seq = []
        mems = tuple()
        # Compute sequence of 128 chars
        for i in range(128)
            ret = self.model(data, target, *mems)
            # Get probabilities of next token
            probs, mems[i] = ret[0], ret[1:]
            # Sample token with temeperature 0.1
            next_tok = gumbel_softmax(probs, 0.1)
            # Shift and concatenate next token
            data = data[:, 1:]
            data = data.concatenate(next_tok, axis=1)
            print(next_tok)

        return data

class Discriminator(nn.Module, transformer):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        transformer = torch.load(transformer, torch.device('cpu'))
        model = [   transformer,
        nn.Linear(128, 50),
        nn.LeakyRelu(),
        nn.Linear(50, 2)
        nn.Softmax(2)]


        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
