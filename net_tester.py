# %%
import pandas as pd
import pickle
from SMRNN import SMRNN

# Load the csv data as a pandas dataframe
google_stock_df = pd.read_csv('datasets/EURUSDX.csv')
test_data = google_stock_df[['Open','Close','High','Low']][-1000:]/10
norm_train_data = ((test_data-test_data.min())/(test_data.max()-test_data.min()))
test_data = norm_train_data.copy()
f = open('nets/net5.obj', 'rb')
net: SMRNN = pickle.load(f)

# %%
import torch
import random
from matplotlib import pyplot
# train the network (in this case only with the open price)
tests = 200
# the batch sizes will be random to let the model to learn in diferent lengths
max_test_size = 40
min_test_size = 40
losses = []
net.float()
for epoch in range(tests):
    test_batch_size = 40
    test_index = random.randint(0,len(test_data) - max_test_size - 5)
    for batch_num in range(test_batch_size):
        inpt = torch.tensor(test_data.iloc[test_index+batch_num].values)
        output = net(inpt.float())
    expected_out = torch.tensor(float(test_data.iloc[test_index+test_batch_size]['Open']))
    loss = torch.abs(expected_out - output)
    # loss = loss_func(output,expected_out)
    print(loss)
    losses.append(float(loss))
    print(f'Predicted: {float(output)}, actual: {float(expected_out)}')
    net.clean()

pyplot.plot(list(range(tests)),losses)
pyplot.show()
print(torch.tensor(losses).mean())
    
