# %%
import pandas as pd

# Load the csv data as a pandas dataframe
google_stock_df = pd.read_csv('datasets/GOOGL.csv')

open_price_data = google_stock_df['Open']
# normalized_open_data = (open_price_data-open_price_data.min())/(open_price_data.max()-open_price_data.min())*10
normalized_open_data = open_price_data
# Separate into train a test data
train_data = normalized_open_data[:-1000]
test_data = normalized_open_data[-1000:]



# %%
from SMRNN import SMRNN
from torch import optim
import torch.nn as nn
# Create the network optimizer and loss function
net = SMRNN(inputs = 1, outputs = 1, inner_state_size = 2,net1_inner_topology = [4,5,2], net2_inner_topology = [10],net3_inner_topology = [7])
optimizer = optim.Adam(net.parameters(),lr=1e-1)
scheduler = optim.lr_scheduler.LinearLR(optimizer,0.1,0.01)
loss_func = nn.MSELoss()



# %%
import torch
#detect gpu and move the net there
torch.cuda.is_available()

# %%
import random
from matplotlib import pyplot
# train the network (in this case only with the open price)
epochs = 500
# the batch sizes will be random to let the model to learn in diferent lengths
max_batch_size = 7
min_batch_size = 5
losses = []
for epoch in range(epochs):
    batch_size = random.randint(5,20)
    train_index = random.randint(0,train_data.size - max_batch_size - 20)
    # Create the batch and normalize it
    batch = train_data[train_index:train_index+batch_size].copy()
    batch_range = batch.max()-batch.min()
    batch = (batch-batch.min())/batch_range
    for batch_num in range(batch_size):
        inpt = torch.tensor([float(batch[train_index])])
        output = net(inpt)
    expected_out = torch.tensor([float(train_data[train_index+batch_size])])
    normalized_expected_out = (expected_out-batch.min())/batch_range
    loss = torch.abs(output - normalized_expected_out)
    # loss = loss_func(output,expected_out)
    print(loss)
    print(f'Predicted: {float((output*batch_range)+batch.min() )}, actual: {float(expected_out)}')
    losses.append(float(loss))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 99:
        scheduler.step()
    net.clean()

pyplot.plot(list(range(epochs - 10)),losses[10:])
pyplot.show()
    


