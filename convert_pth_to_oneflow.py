import numpy as np
import math, json
import torch
import torch.nn as nn
from torch.nn import functional as F

import src.utils
from src.model import GPT, GPTConfig
import oneflow as flow
import shutil
import os




SAVE_PATH = './model/'
DATA_NAME = 'wangwen'
RUN_DEVICE = 'gpu' # gpu 或 dml 或 cpu
MODEL_NAME = './model/wangwen' % DATA_NAME # 模型名
WORD_NAME = './model/wangwen' % DATA_NAME # 这个也修改


ctx_len = 512    # 模型关注的句子长度
n_layer = 12
n_head = 12
n_embd = n_head * 64
n_attn = n_embd
n_ffn = n_embd

with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
    word_table = json.load(result_file)   

vocab_size = len(word_table)
train_dataset = lambda: None
train_dataset.stoi = {v: int(k) for k, v in word_table.items()}
train_dataset.itos = {int(k): v for k, v in word_table.items()}
UNKNOWN_CHAR = train_dataset.stoi['0']


if RUN_DEVICE == 'dml':
    import onnxruntime as rt
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_mem_pattern = False
    rt_session = rt.InferenceSession(MODEL_NAME + '.onnx', sess_options=sess_options, providers=['DmlExecutionProvider'])
    rt_session.set_providers(['DmlExecutionProvider'])
else:
    model = GPT(GPTConfig(vocab_size, ctx_len, n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_attn=n_attn, n_ffn=n_ffn))
    m2 = torch.load(MODEL_NAME + '.pth', map_location='cpu').state_dict()
    for i in range(n_layer):
        prefix = f'blocks.{i}.attn.'
        time_w = m2[prefix + 'time_w']
        time_alpha = m2[prefix + 'time_alpha']
        time_beta = m2[prefix + 'time_beta']
        
        TT = ctx_len
        T = ctx_len
        
        w = F.pad(time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:]
        w = w[:, :T, :T] * time_alpha[:, :, :T] * time_beta[:, :T, :]
        
        m2[prefix + 'time_ww'] = w
        del m2[prefix + 'time_w']
        del m2[prefix + 'time_alpha']
        del m2[prefix + 'time_beta']

    for key, value in m2.items():
        val = value.detach().cpu().numpy()
        m2[key] = val
    model.load_state_dict(m2)
    
    flow.save(model.state_dict(), SAVE_PATH)

