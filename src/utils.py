########################################################################################################
# AI人工智障写作 - https://github.com/BlinkDL/AI-Writer
########################################################################################################

import random
import numpy as np

import oneflow as torch
import oneflow.nn as nn
from oneflow.nn import functional as F

def to_float(x):
    return x.cpu().detach().numpy().flatten()[0].astype(float)

def sample_logits(logits, pos, temperature=1.0, top_p=None):
    logits = logits[0][pos, :]
    probs = F.softmax(logits, dim=-1)

    if top_p is not None:
        out = probs.clone()
        sorted_probs, _ = torch.sort(out, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[int(np.argmax(cumulative_probs > top_p))].cpu().numpy())

        probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    
    p = probs.numpy().astype(np.float64)
    p /= p.sum()
    ix = np.random.choice(np.arange(probs.shape[0]), size=1, p=p)

    # sample = np.random.multinomial(n=1, pvals=p)
    # ix = np.argmax(sample)

    return int(ix)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda:
        torch.cuda.manual_seed_all(seed)
