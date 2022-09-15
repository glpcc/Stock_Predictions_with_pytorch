# %%
import pandas as pd

# Load the csv data as a pandas dataframe
google_stock_df = pd.read_csv('datasets/EURUSDX.csv')
train_data = google_stock_df[['Open','Close','High','Low']][:-1000]
test_data = google_stock_df[['Open','Close','High','Low']][-1000:]

norm_train_data = ((train_data-train_data.min())/(train_data.max()-train_data.min()))

def unnormalize(x, field = 'Open'):
    return (x*(train_data[field].max()-train_data[field].min())) + train_data[field].min()
# %%
from SMRNN import SMRNN
from torch import optim
import torch.nn as nn
# Create the network optimizer and loss function
net = SMRNN(inputs = 4, outputs = 1, inner_state_size = 8,net1_inner_topology = [15,20,20,5,2], net2_inner_topology = [15,20,40,70],net3_inner_topology = [15,10,8,5])
optimizer = optim.Adam(net.parameters(),lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.9)
loss_func = nn.L1Loss()


# %%
import torch
import random
from matplotlib import pyplot
import math
# train the network (in this case only with the open price)
epochs = 2000
# the batch sizes will be random to let the model to learn in diferent lengths
max_batch_size = 40
min_batch_size = 40
losses = []
net.double()
for epoch in range(epochs):
    batch_size = 40
    train_index = random.randint(0,len(train_data)- max_batch_size - 20)
    for batch_num in range(batch_size):
        inpt = torch.tensor(norm_train_data.iloc[train_index+batch_num].values)
        output = net(inpt.double())
    if math.isnan(float(output)):
        # reset network
        net = SMRNN(inputs = 4, outputs = 1, inner_state_size = 8,net1_inner_topology = [15,20,40,20,5,2], net2_inner_topology = [15,20,50,70],net3_inner_topology = [15,10,8,5])
        continue
    expected_out = torch.tensor(float(norm_train_data['Open'][train_index+batch_size]))
    # loss = torch.abs(expected_out - output)
    loss = loss_func(output,expected_out)
    losses.append(float(loss))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'Predicted: {unnormalize(float(output)):.3f}, actual: {unnormalize(float(expected_out)):.3f}, loss: {float(loss):.3f}')
    if epoch % 50 == 51:
        scheduler.step()
    net.clean()

pyplot.plot(list(range(epochs - 10)),losses[10:])
pyplot.show()
    
# Save net 
saved_nets = 5
import pickle
f = open(f'nets/net{saved_nets}.obj','wb')
pickle.dump(net,f)

