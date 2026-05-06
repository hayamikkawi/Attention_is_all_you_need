from encoder_unit import EncoderUnit
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, 
                 units_count: int,
                 d_model: int,
                 heads: int, 
                 d_ff: int):
        super().__init__()
        self.units = nn.ModuleList([EncoderUnit(d_model, heads, d_ff) for _ in range(units_count)])    
    
    def forward(self, input: torch.Tensor): 
        output = input
        for unit in self.units:
            output = unit(output)
        return output

        