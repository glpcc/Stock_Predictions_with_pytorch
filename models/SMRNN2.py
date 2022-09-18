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
        
        self.net3_inner_topology = net3_inner_topology
        self.net2_inner_topology = net2_inner_topology
        self.net1_inner_topology = net1_inner_topology

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

        self.activation_funtion = nn.LeakyReLU()
        self.dropout_layer = nn.Dropout(0)

        self.state = torch.zeros(inner_state_size)
        self.prev_output = torch.zeros(self.outputs)
        self.prev_weights = torch.zeros(sum((net3_inner_topology[j]*net3_inner_topology[j+1] for j in range(len(net3_inner_topology)-1))))
        self.prev_bias = torch.zeros(sum(net3_inner_topology[1:]))

    def forward(self, inputs):
        inpt = torch.cat((inputs,self.prev_output,self.state))
        # Calculate net2 output
        x2 = inpt
        x2 = torch.tanh(self.net2(x2))/2
        weights = x2[:self.prev_weights.size(0)]
        bias = x2[self.prev_weights.size(0):]
        
        self.prev_weights = weights
        self.prev_bias = bias
        # Calculate state (net3 output)
        x3 = inpt
        weight_ind = 0
        bias_ind = 0
        for j in range(1,len(self.net3_inner_topology)):
            # Calculate the bias and weight change
            layer_weights = torch.reshape(self.prev_weights[weight_ind:weight_ind+(self.net3_inner_topology[j]*self.net3_inner_topology[j-1])],(self.net3_inner_topology[j],self.net3_inner_topology[j-1]))
            layer_bias = self.prev_bias[bias_ind:bias_ind+self.net3_inner_topology[j]]
            # Calculate the output
            x3 = x3 @ layer_weights.t() + layer_bias
            if j != len(self.net3_inner_topology) - 1:
                x3 = self.activation_funtion(x3)
                x3 = self.dropout_layer(x3)
            weight_ind += self.net3_inner_topology[j]*self.net3_inner_topology[j-1]
            bias_ind += self.net3_inner_topology[j]

        self.state = x3

        # Calculate net1 output
        x1 = inpt
        x1 = self.net1(x1)
        self.prev_output = x1

        return x1

    def clean(self):
        self.state = self.state.detach()*0
        self.prev_output = self.prev_output.detach()*0
        self.prev_weights = self.prev_weights.detach()*0
        self.prev_bias = self.prev_bias.detach()*0

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.state = self.state.to(*args, **kwargs) 
        self.prev_output = self.prev_output.to(*args, **kwargs) 
        self.prev_weights = self.prev_weights.to(*args, **kwargs) 
        self.prev_bias = self.prev_bias.to(*args, **kwargs) 
        return self