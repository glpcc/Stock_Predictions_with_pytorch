import torch
from torch import nn
from models.layers.SMLSTM import SMLSTM

class Hybrid(nn.Module):
    def __init__(self, inputs: int, outputs: int, lstm_hidden_units: int, gru_hidden_units: int, smlstm_hidden_units: int) -> None:
        '''
            Hybrid between GRU, LSTM and SMLSTM
        '''
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

        self.lstm_hidden_units = lstm_hidden_units
        self.lstm_h_t = torch.zeros(1,lstm_hidden_units)
        self.lstm_c_t = torch.zeros(1,lstm_hidden_units)

        self.gru_hidden_units = gru_hidden_units
        self.gru_h_t = torch.zeros(1,gru_hidden_units)

        self.smlstm_hidden_units = smlstm_hidden_units
        self.smlstm_h_t = torch.zeros(1,smlstm_hidden_units)
        self.smlstm_c_t = torch.zeros(1,smlstm_hidden_units)

        # Layers in order
        self.gru_layer = nn.GRU(input_size=inputs, hidden_size = gru_hidden_units,dropout=0)
        self.act1_layer = nn.LeakyReLU()
        self.lstm_layer = nn.LSTM(input_size=gru_hidden_units, hidden_size = lstm_hidden_units,dropout=0)
        self.act2_layer = nn.LeakyReLU()
        self.smlstm_layer = SMLSTM(input_size=lstm_hidden_units, hidden_size = smlstm_hidden_units,dropout=0)
        self.act3_layer = nn.LeakyReLU()
        self.lin_layer = nn.Linear(smlstm_hidden_units, outputs)

    def forward(self, inputs):
        # Adapt the input
        inputs = torch.reshape(inputs, (1,inputs.size(0)))
        x, self.gru_h_t = self.gru_layer(inputs,self.gru_h_t)
        # x = self.act1_layer(x)

        x, (self.lstm_h_t,self.lstm_c_t) = self.lstm_layer(x,(self.lstm_h_t, self.lstm_c_t))
        # x = self.act2_layer(x)

        x, self.smlstm_c_t , self.smlstm_h_t = self.smlstm_layer(x,self.smlstm_c_t,self.smlstm_h_t)
        # x = self.act3_layer(x)
        x = self.lin_layer(x)
        return x

    def clean(self):
        self.lstm_h_t = torch.zeros(1,self.lstm_hidden_units, device=self.lstm_h_t.device)
        self.lstm_c_t = torch.zeros(1,self.lstm_hidden_units, device=self.lstm_c_t.device)

        self.gru_h_t = torch.zeros(1,self.gru_hidden_units, device=self.gru_h_t.device)

        self.smlstm_h_t = torch.zeros(1,self.smlstm_hidden_units, device=self.smlstm_h_t.device)
        self.smlstm_c_t = torch.zeros(1,self.smlstm_hidden_units, device=self.smlstm_c_t.device)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.lstm_h_t = self.lstm_h_t.to(*args, **kwargs)
        self.lstm_c_t = self.lstm_c_t.to(*args, **kwargs)

        self.gru_h_t = self.gru_h_t.to(*args, **kwargs)

        self.smlstm_h_t = self.smlstm_h_t.to(*args, **kwargs)
        self.smlstm_c_t = self.smlstm_c_t.to(*args, **kwargs)
        return self