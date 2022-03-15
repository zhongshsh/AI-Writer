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

SAVE_PATH = './model/wangwen'

if os.path.isfile(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)


# src.utils.set_seed(42) # 是否固定随机数（固定后每次运行的生成结果都一样）

#
# 需 pytorch 1.9.x 及以上版本
#
# gpu：只支持 nvidia 显卡，速度最快，需 cuda+cudnn
# dml：支持 amd / intel / nvidia 显卡，需不同模型，需 pip install onnxruntime-directml 然后在 run.py 和 server.py 设置为 dml 模式
# cpu：没显卡就选它，但也用 nvidia 卡的模型
DATA_NAME = 'wangwen'
NUM_OF_RUNS = 3# 写多少遍
LENGTH_OF_EACH = 600 # 每次写多少字
top_p = 0.75 # 这个的范围是 0 到 1。越大，变化越多。越小，生成效果越规矩。自己试试 0 和 0.5 和 1.0 的效果就知道了
top_p_newline = 0.9

RUN_DEVICE = 'gpu' # gpu 或 dml 或 cpu
if DATA_NAME == 'wangwen':
    MODEL_NAME = '../RWKV-LM/saved_model_pth/%s/wangwen' % DATA_NAME # 模型名
    WORD_NAME = '../RWKV-LM/saved_model_pth/%s/wangwen' % DATA_NAME # 这个也修改

    ctx_len = 512    # 模型关注的句子长度
    n_layer = 12
    n_head = 12

else:
    MODEL_NAME = '../RWKV-LM/saved_model_pth/%s/trained-100' % DATA_NAME # 模型名
    WORD_NAME = '../RWKV-LM/saved_model_pth/%s/vocab' % DATA_NAME # 这个也修改

    ctx_len = 256    # 模型关注的句子长度
    n_layer = 6
    n_head = 8

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

