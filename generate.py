import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from tx1_midi import tx1_to_midi
import math

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/nesmdb_tx1/test/002_1943_TheBattleofMidway_00_01Title.tx1.txt', help='location of data')
parser.add_argument('--model', type=str, help='location of model')
parser.add_argument('--type', type=str, default='transformer', help='location of model')
parser.add_argument('--out', type=str, default='outs', help='where to save to')

opt = parser.parse_args()

if opt.type == 'transformer':
    from transformer import GeneratorT
    model = GeneratorT(80)
else:
    from lstm import GeneratorRNN
    model = GeneratorRNN(128, 1, 128, bidirectional=False, dropout=0.1, attn='dot', tied=False)

model.load_state_dict(torch.load(opt.model, map_location='cpu')['genBA'])

print("preparing data...")
with open('tx1_vocab.txt','r') as f: #make a dict mapping indices to TX1 tokens
    tokens = f.readlines()
    tokens = [tok.strip('\n') for tok in tokens]
    idx2tok = {i:tokens[i-1] for i in range(1,631)}
idx2tok[0] = '<eos>'

tok2idx = {tok: idx for idx, tok in idx2tok.items()}
seq = []
seq.append(0)
with open(opt.data, 'r') as midi:
    for tok in midi:
        tok = tok.strip('\n')
        event, value = tok.split('_')[:2]
        if event == 'WT':
            value = int(value)
            if value > 100 and value < 1000:
                value = math.ceil(value / 10.0) * 10
            elif value > 1000 and value < 10000:
                value = math.ceil(value / 100.0) * 100
            elif value > 10000 and value < 100000:
                value = math.ceil(value / 1000.0) * 1000
            elif value > 100000:
                value = 100000
            tok = f'WT_{value}'
        print(tok)
        seq.append(tok2idx[tok])
seq.append(0)

seq = torch.tensor(seq)
tensor = torch.zeros(40 * (seq.size(0) // 40 + 1))
tensor[:seq.size(0)] = seq
tensor = tensor.reshape(40, -1).long()

tensor = F.one_hot(tensor, 631).float()
with torch.no_grad():
    model.eval()
    output = model.forward(tensor, 0.01)

output = torch.argmax(output, dim=-1)
output = output.flatten().tolist()
print(output)
with open(f'{opt.out}.tx1.txt', 'w') as tx1:
    for tok in output:
        tx1.write(f'{idx2tok[tok]}\n')

with open(f'{opt.out}.tx1.txt', 'r') as tx1:
    tokens = tx1.read()

midi = tx1_to_midi(tokens)
midi.write(f'{opt.out}.mid')
