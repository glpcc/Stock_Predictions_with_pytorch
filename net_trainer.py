# %%
from cmath import isnan
from tokenize import Double
import pandas as pd

# Load the csv data as a pandas dataframe
google_stock_df = pd.read_csv('datasets/GOOGL.csv')
train_data = google_stock_df[['Open','Close','High','Low']][:-1000]/100
test_data = google_stock_df[['Open','Close','High','Low']][-1000:]/100

# %%
from SMRNN import SMRNN
from torch import optim
import torch.nn as nn
# Create the network optimizer and loss function
net = SMRNN(inputs = 4, outputs = 1, inner_state_size = 8,net1_inner_topology = [15,20,40,20,5,2], net2_inner_topology = [15,20,40,70],net3_inner_topology = [15,10,8,5])
optimizer = optim.Adam(net.parameters(),lr=1e-2)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.8)
loss_func = nn.MSELoss()


# %%
import torch
import random
from matplotlib import pyplot
import math
# train the network (in this case only with the open price)
epochs = 3000
# the batch sizes will be random to let the model to learn in diferent lengths
max_batch_size = 150
min_batch_size = 100
losses = []
net.float()
for epoch in range(epochs):
    batch_size = random.randint(min_batch_size,max_batch_size)
    train_index = random.randint(0,len(train_data)- max_batch_size - 20)
    for batch_num in range(batch_size):
        inpt = torch.tensor(train_data.iloc[train_index+batch_num].values)
        output = net(inpt.float())
    if math.isnan(float(output)):
        # reset network
        net = SMRNN(inputs = 4, outputs = 1, inner_state_size = 8,net1_inner_topology = [15,20,40,20,5,2], net2_inner_topology = [15,20,50,70],net3_inner_topology = [15,10,8,5])
        continue
    expected_out = torch.tensor(float(train_data['Open'][train_index+batch_size]))
    loss = torch.abs(expected_out - output)
    # loss = loss_func(output,expected_out)
    losses.append(float(loss))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'Predicted: {float(output):.3f}, actual: {float(expected_out):.3f}, loss: {float(loss):.3f}')
    if epoch % 100 == 99:
        scheduler.step()
    net.clean()

pyplot.plot(list(range(epochs - 10)),losses[10:])
pyplot.show()
    
# Save net 
saved_nets = 3
import pickle
f = open(f'nets/net{saved_nets}.obj','wb')
pickle.dump(net,f)

