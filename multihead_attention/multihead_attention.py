import torch 
import torch.nn as nn
from scaled_dot_prod_attention import scaled_dot_prod_attention

class MultiHeadAttention(nn.Module):
    # Q: (batch_size, seq_len, d_model)
    # K: (batch_size, seq_len, d_model)
    # V: (batch_size, seq_len, d_model)
    def forward(self, Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor, 
                mask: torch.Tensor=None):
        # linear project the tensors 
        Q = self.W_Q(Q) # (batch_size, seq_len, d_model)
        K = self.W_K(K) # (batch_size, seq_len, d_model)
        V = self.W_V(V) # (batch_size, seq_len, d_model)
        # split to heads
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        d_k = self.d_model // self.h
        d_v = d_k
        Q = torch.reshape(Q, (batch_size, seq_len, self.h, d_k)) # (batch_size, seq_len, h, d_k)
        K = torch.reshape(K, (batch_size, seq_len, self.h, d_k)) # (batch_size, seq_len, h, d_k)
        V = torch.reshape(V, (batch_size, seq_len, self.h, d_v)) # (batch_size, seq_len, h, d_v)
        Q = Q.transpose(1, 2) # (batch_size, h, seq_len, d_k)
        K = K.transpose(1, 2) # (batch_size, h, seq_len, d_k)
        V = V.transpose(1, 2) # (batch_size, h, seq_len, d_v)
        # call scaled dot prod. attention on each head
        out = scaled_dot_prod_attention(Q, K, V, mask=mask) # (batch_size, h, seq_len, d_v)
        # concatenate all heads together
        out = out.transpose(1, 2)
        out = torch.reshape(out, (batch_size, seq_len, self.h*d_v)) # (batch_size, seq_len, d_model)
        out = self.W_O(out) # (batch_size, seq_len, d_model)
        return out

    def __init__(self, d_model: int, heads: int):
        super().__init__() 
        self.h = heads
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model)   # projects Q
        self.W_K = nn.Linear(d_model, d_model)   # projects K
        self.W_V = nn.Linear(d_model, d_model)   # projects V
        self.W_O = nn.Linear(d_model, d_model)   # final output projection