# %%
import pandas as pd
import pickle
import random
from matplotlib import pyplot
import torch


# Load the gpu (in my case it actually runs slower so i turned it off)
if torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cpu')


    
# Load the csv data as a pandas dataframe
eurusd_stock_df = pd.read_csv('datasets/EURUSDX.csv')
train_data = eurusd_stock_df[['Open','Close','High','Low']][:-1000]
test_data = eurusd_stock_df[['Open','Close','High','Low']][-1000:]

norm_test_data = ((test_data-test_data.min())/(test_data.max()-test_data.min()))

norm_test_data = torch.tensor(norm_test_data.values)
norm_test_data = norm_test_data.to(device)
norm_test_data = norm_test_data.float()

# Utility function
def unnormalize(x, field = 'Open'):
    return (x*(test_data[field].max()-test_data[field].min())) + test_data[field].min()

f = open('nets/FGRNN_best_net_EURUSD_20.obj', 'rb')
net = pickle.load(f)
net.to(device)

batch_size = 20
real_open_prices = test_data['Open'].values
predicted_prices = []

net.eval()
epochs = len(norm_test_data) - 2*batch_size - 1
for epoch in range(epochs):
    test_index = batch_size + epoch
    for batch_num in range(batch_size):
        inpt = norm_test_data[test_index+batch_num]
        output = net(inpt)
    predicted_prices.append(unnormalize(output))
    net.clean()

pyplot.plot(list(real_open_prices[batch_size:]))
pyplot.plot(torch.tensor(predicted_prices))
pyplot.legend()
pyplot.show()

    
