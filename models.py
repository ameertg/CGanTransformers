import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

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
    return torch.nonzero((y_hard - y).detach() + y)

class Generator(nn.Module):
    def __init__(self, transformer):
        super(Generator, self).__init__()

        # Load transformer
        self.model = torch.load(transformer, torch.device('cpu'))

    def forward(self, x, bptt):
        seq = []
        mems = tuple()
        # Compute sequence of 128 tokens
        for i in range(bptt):
            ret = self.model._forward(x, *mems)
            # Get probabilities of next token
            probs, mems = ret[0], ret[1:]
            # Sample token with temeperature 0.1
            next_tok = gumbel_softmax(probs[-1, :, :], 0.1)[:, 1].unsqueeze(0)
            # Shift and concatenate next token
            x = x[1:, :]
            x = torch.cat([x, next_tok], 0)

        return x

class Discriminator(nn.Module):
    def __init__(self, transformer):
        super(Discriminator, self).__init__()

        self.transformer = torch.load(transformer, torch.device('cpu'))
        model = [nn.Linear(512, 50),
        nn.ReLU(),
        nn.Linear(50, 1)]


        self.model = nn.Sequential(*model)
        self.final = nn.Linear(12, 1)

    def forward(self, x):
        mems = tuple()
        x = self.transformer._forward(x, *mems)[0]
        x = x.transpose(0,1)
        x = self.model(x).squeeze(2)
        x = self.final(x)
        return torch.sigmoid(x)
