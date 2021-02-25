from collections import Counter

import glob

fps = glob.glob('*/*.tx1.txt')
print(len(fps))

vocab = set()
waits = Counter()
for fp in fps:
  with open(fp, 'r') as f:
    events = f.read().strip().splitlines()
  for e in events:
    if e[:2] == 'WT':
      waits[int(e[3:])] += 1
    else:
      vocab.add(e)

for v in sorted(list(vocab)):
  print v
#print waits
