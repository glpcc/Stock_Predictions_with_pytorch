# %%
import torch
import pandas as pd
from SMRNN import SMRNN
from RNN import SMRNN2
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
net = SMRNN2(inputs = 4, outputs = 1, inner_state_size = 20,net1_inner_topology = [40,60,100,50,20,10,5], net2_inner_topology = [30,40,30,20,20])
optimizer = optim.Adam(net.parameters(),lr=1e-3)
scheduler = optim.lr_scheduler.LambdaLR(optimizer,lambda epoch: 0.5**epoch)
loss_func = nn.L1Loss()

# train the network (in this case only with the open price)
epochs = 3000
# the batch sizes will be random to let the model to learn in diferent lengths
max_batch_size = 40
min_batch_size = 40
losses = []
# net.double()
net.to(device)
for epoch in range(epochs):
    batch_size = 40
    train_index = random.randint(0,len(train_data)- max_batch_size - 20)

    for batch_num in range(batch_size):
        inpt = norm_train_data[train_index+batch_num]
        output = net(inpt)
    
    if math.isnan(float(output)):
        # reset network
        net = SMRNN2(inputs = 4, outputs = 1, inner_state_size = 8,net1_inner_topology = [15,20,40,20,5,2], net2_inner_topology = [15,20,50,70],net3_inner_topology = [15,10,8,5])
        net.to(device)
        continue
    
    expected_out = norm_train_data[train_index+batch_size][0]
    # loss = torch.abs(expected_out - output)
    loss = loss_func(output,expected_out)
    
    losses.append(float(loss))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'Predicted: {unnormalize(float(output)):.3f}, actual: {unnormalize(float(expected_out)):.3f}, loss: {float(loss):.3f}')
    if epoch % 300 == 299:
        scheduler.step()
        print(scheduler.get_lr())
    net.clean()

pyplot.plot(list(range(epochs - 10)),losses[10:])
pyplot.show()
    
# Save net 
saved_nets = 8
import pickle
f = open(f'nets/net{saved_nets}.obj','wb')
pickle.dump(net,f)

