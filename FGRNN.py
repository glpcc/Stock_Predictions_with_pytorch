import torch
from torch import nn


class FGRNN(nn.Module):
    def __init__(self, inputs: int, outputs: int, inner_state_size: int, net1_inner_topology: list[int], net2_inner_topology: list[int], net3_inner_topology: list[int]) -> None:
        '''
            See the pdf with the diagram to understand the net nomenclature
        '''
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.inner_state_size = inner_state_size

        net1_inner_topology = [inputs + outputs + inner_state_size] + net1_inner_topology + [outputs]
        net2_inner_topology = [inputs + outputs + inner_state_size] + net1_inner_topology + [inner_state_size]
        net3_inner_topology = [inputs + outputs + inner_state_size] + net1_inner_topology + [inner_state_size]
        
        dropout_prob = 0
        self.net1 = nn.Sequential()
        for i in range(len(net1_inner_topology)-1):
            self.net1.add_module(f"Linear({net1_inner_topology[i]},{net1_inner_topology[i+1]})", nn.Linear(net1_inner_topology[i],net1_inner_topology[i+1]))
            if i != len(net1_inner_topology)-2:
                self.net1.add_module(f"ReLU {i}", nn.ReLU())
                self.net1.add_module(f"Dropout {i}", nn.Dropout(dropout_prob))

        self.net2 = nn.Sequential()
        for i in range(len(net2_inner_topology)-1):
            self.net2.add_module(f"Linear({net2_inner_topology[i]},{net2_inner_topology[i+1]})", nn.Linear(net2_inner_topology[i],net2_inner_topology[i+1]))
            if i != len(net2_inner_topology)-2:
                self.net2.add_module(f"ReLU {i}", nn.ReLU())
                self.net2.add_module(f"Dropout {i}", nn.Dropout(dropout_prob))

        self.net3 = nn.Sequential()
        for i in range(len(net3_inner_topology)-1):
            self.net3.add_module(f"Linear({net3_inner_topology[i]},{net3_inner_topology[i+1]})", nn.Linear(net3_inner_topology[i],net3_inner_topology[i+1]))
            if i != len(net3_inner_topology)-2:
                self.net3.add_module(f"ReLU {i}", nn.ReLU())
                self.net3.add_module(f"Dropout {i}", nn.Dropout(dropout_prob))

        self.state = torch.zeros(inner_state_size)
        self.prev_output = torch.zeros(self.outputs)

    def forward(self, inputs):
        inpt = torch.cat((inputs,self.prev_output,self.state))
        # Calculate net2 output ; the state
        x2 = inpt
        x2 = self.net2(x2)

        # Calculate net3 output (forget gate)
        x3 = inpt
        x3 = torch.tanh(self.net3(x3))
        self.state = torch.tanh(x2)*x3

        # Calculate net1 output
        x1 = inpt
        x1 = self.net1(x1)
        self.prev_output = x1

        return x1

    def clean(self):
        self.state = self.state.detach()*0
        self.prev_output = self.prev_output.detach()*0

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.state = self.state.to(*args, **kwargs) 
        self.prev_output = self.prev_output.to(*args, **kwargs) 
        return self