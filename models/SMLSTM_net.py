from turtle import forward
import torch
from torch import nn
from models.layers.SMLSTM import SMLSTM

class SMLSTM_net(nn.Module):
    def __init__(self,inputs: int, outputs: int, hidden_units: int) -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.smlstm_layer =  SMLSTM(inputs,hidden_units,dropout= 0)
        self.linear_layer = nn.Linear(hidden_units,outputs)
        self.h_t = torch.zeros(hidden_units)
        self.c_t = torch.zeros(hidden_units)

    def forward(self, input):
        out, self.c_t , self.h_t = self.smlstm_layer(input,self.c_t,self.h_t)
        out = self.linear_layer(out)
        return out
    
    def clean(self):
        self.h_t = torch.zeros(self.hidden_units, device=self.h_t.device)
        self.c_t = torch.zeros(self.hidden_units, device=self.c_t.device)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.h_t = self.h_t.to(*args, **kwargs)
        self.c_t = self.c_t.to(*args, **kwargs)
        return self
    
    def show_sm_out(self):
        return self.smlstm_layer.show_sm_out()