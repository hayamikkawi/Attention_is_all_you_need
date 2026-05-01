import torch
import torch.nn.functional as F

# Q: (batch_size, seq_len, d_k)
# K: (batch_size, seq_len, d_k)
# V: (batch_size, seq_len, d_v)
def scaled_dot_prod_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    k_trans = K.transpose(-2, -1) #(batch_size, d_k, seq_len)
    scores = torch.matmul(Q, k_trans) #(batch_size, seq_len, seq_len)
    d_k = Q.size(-1)
    scaled = scores / (d_k ** 0.5) #(batch_size, seq_len, seq_len)
    if mask is not None: 
        scaled = scaled.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scaled, dim=-1) #(batch_size, seq_len, seq_len)
    out = torch.matmul(weights, V) #(batch_size, seq_len, d_v)
    return out