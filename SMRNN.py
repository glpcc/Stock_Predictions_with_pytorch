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
                self.net1.add_module(f"ReLU {i}", nn.ReLU())

        self.net2 = nn.Sequential()
        for i in range(len(net2_inner_topology)-1):
            self.net2.add_module(f"Linear({net2_inner_topology[i]},{net2_inner_topology[i+1]})", nn.Linear(net2_inner_topology[i],net2_inner_topology[i+1]))
            if i != len(net2_inner_topology)-2:
                self.net2.add_module(f"ReLU {i}", nn.ReLU())

        self.net3_weigths = [(torch.rand(net3_inner_topology[i+1],net3_inner_topology[i])-0.5)/1000 for i in range(len(net3_inner_topology)-1)]
        self.net3_biases = [torch.zeros(i) for i in net3_inner_topology[1:]]
        self.activation_funtion = nn.ReLU()

        self.state = torch.zeros(inner_state_size)
        self.prev_output = torch.zeros(self.outputs)
        self.prev_weight_change = torch.zeros(sum((net3_inner_topology[j]*net3_inner_topology[j+1] for j in range(len(net3_inner_topology)-1))))
        self.prev_bias_change = torch.zeros(sum(net3_inner_topology[1:]))

    def forward(self, inputs):
        inpt = torch.cat((inputs,self.prev_output,self.state))
        # Calculate net2 output
        x2 = inpt
        x2 = torch.tanh(self.net2(x2))*2
        weight_change = x2[:self.prev_weight_change.size(0)]
        bias_change = x2[self.prev_weight_change.size(0):]

        # Change net3 weights and biases
        i = 0
        for j in range(len(self.net3_weigths)):
            self.net3_weigths[j] = self.net3_weigths[j] + torch.reshape(self.prev_weight_change[i:i+(self.net3_weigths[j].size(0)*self.net3_weigths[j].size(1))],self.net3_weigths[j].size())
            i += self.net3_weigths[j].size(0)*self.net3_weigths[j].size(1)

        i = 0
        for j in range(len(self.net3_biases)):
            self.net3_biases[j] = self.net3_biases[j] + self.prev_bias_change[i:i+self.net3_biases[j].size(0)]
            i += self.net3_biases[j].size(0)
        
        self.prev_weight_change = weight_change
        self.prev_bias_change = bias_change
        
        # Calculate state (net3 output)
        x3 = inpt
        for i in range(len(self.net3_weigths)):
            x3 = x3 @ self.net3_weigths[i].t() + self.net3_biases[i]
            if i != len(self.net3_weigths) - 1:
                x3 = self.activation_funtion(x3)
        
        self.state = torch.tanh(x3)*2

        # Calculate net1 output
        x1 = inpt
        x1 = self.net1(x1)
        self.prev_output = x1

        return x1

    def clean(self):
        self.state = torch.zeros(self.state.size())
        self.prev_output = torch.zeros(self.outputs)
        self.prev_weight_change = torch.zeros(self.prev_weight_change.size())
        self.prev_bias_change = torch.zeros(self.prev_bias_change.size())
        self.net3_weigths = [(torch.rand(i.size())-0.5)/1000 for i in self.net3_weigths]
        self.net3_biases = [torch.zeros(i.size()) for i in self.net3_biases]
