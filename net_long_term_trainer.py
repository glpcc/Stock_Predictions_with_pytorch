# %%
import torch
import pandas as pd
from models.SMRNN import SMRNN
# from models.SMRNN2 import SMRNN
from models.LSTM import LSTM
from models.SMLSTM_net import SMLSTM_net
from models.Hybrid import Hybrid
from torch import optim
import torch.nn as nn
import random
from matplotlib import pyplot
import math

# Load the gpu (in my case it actually runs slower so i turned it off)
if torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cpu')

# Load the csv data as a pandas dataframe
eurusd_stock_df = pd.read_csv('datasets/EURUSDX.csv')
train_data = eurusd_stock_df[['Open','Close','High','Low']][:-1000]
test_data = eurusd_stock_df[['Open','Close','High','Low']][-1000:]

norm_train_data = ((train_data-train_data.min())/(train_data.max()-train_data.min()))

norm_train_data = torch.tensor(norm_train_data.values)
norm_train_data = norm_train_data.to(device)
norm_train_data = norm_train_data.float()

# Utility function
def unnormalize(x, field = 'Open'):
    return (x*(train_data[field].max()-train_data[field].min())) + train_data[field].min()

# Create the network optimizer and loss function
inputs = 4
outputs = 4
#net = SMRNN(inputs, outputs, inner_state_size=10, net1_inner_topology=[20,30,50,20,5,2],net2_inner_topology=[15,20,40,70,100,200],net3_inner_topology=[15,20,15,10])
net = Hybrid(inputs = 4, outputs = 4, lstm_hidden_units = 50, gru_hidden_units=50, smlstm_hidden_units= 50)
optimizer = optim.Adam(net.parameters(),lr=1e-2)
scheduler = optim.lr_scheduler.LambdaLR(optimizer,lambda epoch: 0.5**epoch)

# train the network (in this case only with the open price)
epochs = 1000
# the batch sizes will be random to let the model to learn in diferent lengths
losses = []
# net.double()
net.to(device)
batch_size = 30
look_ahead_size = 5

for epoch in range(epochs):
    train_index = random.randint(0,len(train_data)- batch_size - 20)
    loss = []

    for i in range(batch_size-look_ahead_size):
        inpt = norm_train_data[train_index + i]
        output = net(inpt)

    inpt = output[0]

    for batch_num in range(look_ahead_size):
        output = net(inpt) 
        inpt = output[0]
        loss.append(torch.abs(output-norm_train_data[train_index+batch_size-look_ahead_size+batch_num]))
    
    loss = torch.cat(loss)
    loss = loss.sum()
    losses.append(float(loss))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'Predicted: {unnormalize(float(output[0][0])):.3f}, actual: {unnormalize(float(norm_train_data[train_index+batch_size][0])):.3f}, loss: {float(loss):.3f}')
    if epoch % 100 == 99:
        scheduler.step()
        print(scheduler.get_lr())
    net.clean()

pyplot.plot(list(range(epochs - 10)),losses[10:])
pyplot.show()
    
# Save net 
saved_nets = 9
import pickle
f = open(f'nets/long_term_learn/net{saved_nets}.obj','wb')
pickle.dump(net,f)

