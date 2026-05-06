import torch.nn as nn
import torch

class FFN(nn.Module): 
    def __init__(self, d_ff:int, d_model: int):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, data: torch.Tensor):
        new_data = self.layer1(data)
        activated = self.activation(new_data)
        output = self.layer2(activated)
        return output
