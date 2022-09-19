import torch 
from torch import nn

class SMLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        def initialize_weight_param(inpt_s, out_s):
            return nn.parameter.Parameter(torch.nn.init.uniform_(torch.empty(out_s,inpt_s),a=1/inpt_s,b=1/inpt_s))

        def initialize_bias_param(inpt_s):
            return nn.parameter.Parameter(torch.nn.init.uniform_(torch.empty(inpt_s),a=1/inpt_s,b=1/inpt_s))

        self.input_size = input_size
        self.hidden_size = hidden_size
        # Forget gate activation vector params
        self.fg_input_weights = initialize_weight_param(input_size,hidden_size)
        self.fg_hidden_weights = initialize_weight_param(hidden_size,hidden_size)
        self.fg_bias = initialize_bias_param(hidden_size)
        # Input gate activation vector params
        self.inpt_input_weights = initialize_weight_param(input_size,hidden_size)
        self.inpt_hidden_weights = initialize_weight_param(hidden_size,hidden_size)
        self.inpt_bias = initialize_bias_param(hidden_size)
        # Output gate activation vector params 
        self.out_input_weights = initialize_weight_param(input_size,hidden_size)
        self.out_hidden_weights = initialize_weight_param(hidden_size,hidden_size)
        self.out_bias = initialize_bias_param(hidden_size)
        # Cell input activation vector params
        self.cinpt_input_weights = initialize_weight_param(input_size,hidden_size)
        self.cinpt_hidden_weights = initialize_weight_param(hidden_size,hidden_size)
        self.cinpt_bias = initialize_bias_param(hidden_size)
        # input weight generator params
        self.wgen_input_input_weights = initialize_weight_param(input_size,hidden_size*input_size)
        self.wgen_input_hidden_weights = initialize_weight_param(hidden_size,hidden_size*input_size)
        self.wgen_input_bias = initialize_bias_param(hidden_size*input_size)
        # hidden weight generator params
        self.wgen_hidden_input_weights = initialize_weight_param(input_size,hidden_size*hidden_size)
        self.wgen_hidden_hidden_weights = initialize_weight_param(hidden_size,hidden_size*hidden_size)
        self.wgen_hidden_bias = initialize_bias_param(hidden_size*hidden_size)
        # Bias generator params
        self.bgen_input_weights = initialize_weight_param(input_size,hidden_size)
        self.bgen_hidden_weights = initialize_weight_param(hidden_size,hidden_size)
        self.bgen_bias = initialize_bias_param(hidden_size)
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        # TODO DROPOUT TO BE IMPLEMENTED

    def forward(self,input, cell_state, hidden_state):
        fg_vec = input @ self.fg_input_weights.t() + hidden_state @ self.fg_hidden_weights.t() + self.fg_bias
        fg_vec = self.sigmoid(fg_vec)

        inptg_vec = input @ self.inpt_input_weights.t() + hidden_state @ self.inpt_hidden_weights.t() + self.inpt_bias
        inptg_vec = self.sigmoid(inptg_vec)

        out_vec = input @ self.out_input_weights.t() + hidden_state @ self.out_hidden_weights.t() + self.out_bias
        out_vec = self.sigmoid(out_vec)

        cinpt = input @ self.cinpt_input_weights.t() + hidden_state @ self.cinpt_hidden_weights.t() + self.cinpt_bias
        cinpt = self.tanh(cinpt)

        gen_inpt_w = input @ self.wgen_input_input_weights.t() + hidden_state @ self.wgen_input_hidden_weights.t() + self.wgen_input_bias
        gen_inpt_w = torch.reshape(gen_inpt_w, (self.hidden_size,self.input_size))

        gen_hidden_w = input @ self.wgen_hidden_input_weights.t() + hidden_state @ self.wgen_hidden_hidden_weights.t() + self.wgen_hidden_bias
        gen_hidden_w = torch.reshape(gen_hidden_w, (self.hidden_size,self.hidden_size))
        
        gen_bias = input @ self.bgen_input_weights.t() + hidden_state @ self.bgen_hidden_weights.t() + self.bgen_bias
        sm_out = self.sigmoid(input @ gen_inpt_w.t() + hidden_state @ gen_hidden_w.t() + gen_bias)
        self.sm_out = sm_out
        new_cell_state = fg_vec*sm_out*cell_state + inptg_vec*cinpt 
        output = out_vec*self.tanh(new_cell_state)

        output = self.dropout(output)
        return output, new_cell_state, output

    def show_sm_out(self):
        return self.sm_out