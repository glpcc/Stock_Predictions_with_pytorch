import torch
from torch import nn


class SMRNN(nn.Module):
    def __init__(self, inputs: int, outputs: int, inner_state_size: int, net1_inner_topology: list[int], net2_inner_topology: list[int], net3_inner_topology: list[int]) -> None:
        '''
            See the pdf with the diagram to understand the net nomenclature
        '''
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.inner_state_size = inner_state_size

        net1_inner_topology = [inputs + outputs + inner_state_size] + net1_inner_topology + [outputs]
        net3_inner_topology = [inputs + outputs + inner_state_size] + net1_inner_topology + [inner_state_size]
        net2_inner_topology = [inputs + outputs + inner_state_size] + net1_inner_topology + [sum((net3_inner_topology[j]*net3_inner_topology[j+1] for j in range(len(net3_inner_topology)-1))) + sum(net3_inner_topology[1:])]
        
        self.net1 = nn.Sequential()
        for i in range(len(net1_inner_topology)-1):
            self.net1.add_module(f"Linear({net1_inner_topology[i]},{net1_inner_topology[i+1]})", nn.Linear(net1_inner_topology[i],net1_inner_topology[i+1]))
            if i != len(net1_inner_topology)-2:
                self.net1.add_module(f"ReLU {i}", nn.LeakyReLU())
                self.net1.add_module(f"Dropout {i}", nn.Dropout(0))

        self.net2 = nn.Sequential()
        for i in range(len(net2_inner_topology)-1):
            self.net2.add_module(f"Linear({net2_inner_topology[i]},{net2_inner_topology[i+1]})", nn.Linear(net2_inner_topology[i],net2_inner_topology[i+1]))
            if i != len(net2_inner_topology)-2:
                self.net2.add_module(f"ReLU {i}", nn.LeakyReLU())
                self.net2.add_module(f"Dropout {i}", nn.Dropout(0))

        self.net3_weigths = [nn.parameter.Parameter((torch.rand(net3_inner_topology[i+1],net3_inner_topology[i])-0.5)/1) for i in range(len(net3_inner_topology)-1)]
        self.net3_biases = [nn.parameter.Parameter(torch.zeros(i)) for i in net3_inner_topology[1:]]
        self.net3_weigths = nn.ParameterList(self.net3_weigths)
        self.net3_biases = nn.ParameterList(self.net3_biases)
        self.activation_funtion = nn.LeakyReLU()
        self.dropout_layer = nn.Dropout(0)

        self.state = torch.zeros(inner_state_size)
        self.prev_output = torch.zeros(self.outputs)
        self.prev_weight_change = torch.zeros(sum((net3_inner_topology[j]*net3_inner_topology[j+1] for j in range(len(net3_inner_topology)-1))))
        self.prev_bias_change = torch.zeros(sum(net3_inner_topology[1:]))

    def forward(self, inputs):
        inpt = torch.cat((inputs,self.prev_output,self.state))
        # Calculate net2 output
        x2 = inpt
        x2 = torch.tanh(self.net2(x2))*5
        weight_change = x2[:self.prev_weight_change.size(0)]
        bias_change = x2[self.prev_weight_change.size(0):]
        
        self.prev_weight_change = weight_change
        self.prev_bias_change = bias_change
        # Calculate state (net3 output)
        x3 = inpt
        weight_ind = 0
        bias_ind = 0
        for j in range(len(self.net3_weigths)):
            # Calculate the bias and weight change
            layer_weight_change = torch.reshape(self.prev_weight_change[weight_ind:weight_ind+(self.net3_weigths[j].size(0)*self.net3_weigths[j].size(1))],self.net3_weigths[j].size())
            layer_bias_change = self.net3_biases[j] + self.prev_bias_change[bias_ind:bias_ind+self.net3_biases[j].size(0)]
            # Calculate the output
            x3 = x3 @ (self.net3_weigths[j]+layer_weight_change).t() + (self.net3_biases[j]+layer_bias_change)
            if j != len(self.net3_weigths) - 1:
                x3 = self.activation_funtion(x3)
                x3 = self.dropout_layer(x3)
            weight_ind += self.net3_weigths[j].size(0)*self.net3_weigths[j].size(1)
            bias_ind += self.net3_biases[j].size(0)

        self.state = torch.tanh(x3)*2

        # Calculate net1 output
        x1 = inpt
        x1 = self.net1(x1)
        self.prev_output = x1

        return x1

    def clean(self):
        self.state = self.state.detach()*0
        self.prev_output = self.prev_output.detach()*0
        self.prev_weight_change = self.prev_weight_change.detach()*0
        self.prev_bias_change = self.prev_bias_change.detach()*0

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.state = self.state.to(*args, **kwargs) 
        self.prev_output = self.prev_output.to(*args, **kwargs) 
        self.prev_weight_change = self.prev_weight_change.to(*args, **kwargs) 
        self.prev_bias_change = self.prev_bias_change.to(*args, **kwargs) 
        return self