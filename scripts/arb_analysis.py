import fileinput
from typing import List
from torch import tensor, FloatTensor
import torch

aspects: List[float] = []

# https://huggingface.co/datasets/Birchlabs/danbooru-aspect-ratios/resolve/main/danbooru-aspect-ratios.csv
with fileinput.input(files=('/home/birch/ml-data/danbooru-aspect-ratios.csv'), encoding='ascii') as f:
  # skip header line
  next(f)
  for line in f:
    aspect = float(line.rstrip('\n'))
    aspects.append(aspect)

device = torch.device('cuda')

a: FloatTensor = tensor(aspects, dtype=torch.float32, device=device)
pass
# I switched from working on this .py, to working on the .ipynb instead