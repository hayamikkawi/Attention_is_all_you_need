import torch
import torch.nn as nn
from multihead_attention.multihead_attention import MultiheadAttention
from FFN import FFN

class EncoderUnit(nn.Module): 
    def __init__(self,
                 d_model: int,
                 heads: int, 
                 d_ff: int):
        super().__init__()
        self.multihead_attention = MultiheadAttention(d_model, heads)
        self.ffn = FFN(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, input: torch.Tensor):
        atten = self.multihead_attention(input, input, input)
        add_normalized1 = self.norm1(atten + input)
        ffned = self.ffn(add_normalized1)
        add_normalized2 = self.norm2(ffned + add_normalized1)
        return add_normalized2

         


