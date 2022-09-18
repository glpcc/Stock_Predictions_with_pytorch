import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden_units: int ) -> None:
        '''
            See the pdf with the diagram to understand the net nomenclature
        '''
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_units = hidden_units
        self.h_t = torch.zeros(1,hidden_units)
        self.c_t = torch.zeros(1,hidden_units)

        self.lstm_layer = nn.LSTM(input_size=inputs, hidden_size = hidden_units,dropout=0)
        self.lin_layer = nn.Linear(hidden_units, outputs)

    def forward(self, inputs):
        # Adapt the input
        inputs = torch.reshape(inputs, (1,inputs.size(0)))
        x, (self.h_t, self.c_t) = self.lstm_layer(inputs, (self.h_t, self.c_t) )
        x = self.lin_layer(x)
        return x

    def clean(self):
        self.h_t = torch.zeros(1,self.hidden_units)
        self.c_t = torch.zeros(1,self.hidden_units)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.h_t = self.h_t.to(*args, **kwargs)
        self.c_t = self.c_t.to(*args, **kwargs)
        return self