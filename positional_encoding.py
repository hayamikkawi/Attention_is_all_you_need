import torch

def positional_encoding(seq_len, d_model): 
    pos = torch.arange(0, seq_len) #(sq_len,)
    pos = torch.reshape(pos, (seq_len, 1)) #(seq_len , 1)
    i = torch.arange(0, d_model, 2) #(1, d_model // 2) (even only)
    denom = 10000**(i / d_model) #(1, d_model // 2)
    angle = pos / denom #(seq_len , d_model // 2)
    sin = torch.sin(angle) #(seq_len , d_model // 2)
    cos = torch.cos(angle) #(seq_len , d_model // 2)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = sin
    pe[:, 1::2] = cos
    return pe #(seq_len , d_model)


    