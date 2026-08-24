# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan T2V 1.3B ------------------------#

t2v_1_3B_train = EasyDict(__name__='Config: Wan T2V 1.3B Train')

# transformer
t2v_1_3B_train.patch_size = (1, 2, 2)
t2v_1_3B_train.dim = 1536
t2v_1_3B_train.in_dim = 36
t2v_1_3B_train.ffn_dim = 8960
t2v_1_3B_train.freq_dim = 256
t2v_1_3B_train.num_heads = 12
t2v_1_3B_train.num_layers = 30
t2v_1_3B_train.window_size = (-1, -1)
t2v_1_3B_train.qk_norm = True
t2v_1_3B_train.cross_attn_norm = True
t2v_1_3B_train.eps = 1e-6
