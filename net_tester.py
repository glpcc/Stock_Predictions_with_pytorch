# %%
import pandas as pd
import pickle
from models.SMRNN import SMRNN
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

f = open('nets/net9.obj', 'rb')
net: SMRNN = pickle.load(f)




# train the network (in this case only with the open price)
tests = 200
# the batch sizes will be random to let the model to learn in diferent lengths
max_test_size = 40
min_test_size = 40
losses = []
actual_losses = []
net.to(device)
for epoch in range(tests):
    test_batch_size = 40
    test_index = random.randint(0,len(norm_test_data) - max_test_size - 5)
    for batch_num in range(test_batch_size):
        inpt = norm_test_data[test_index+batch_num]
        net.eval()
        output = net(inpt)
    expected_out = norm_test_data[test_index+test_batch_size][0]
    loss = torch.abs(expected_out - output)
    # loss = loss_func(output,expected_out)
    print(loss)
    losses.append(float(loss))
    actual_losses.append(abs(unnormalize(expected_out) - unnormalize(output)))
    print(f'Predicted: {float(output)}, actual: {float(expected_out)}')
    net.clean()

pyplot.plot(list(range(tests)),losses)
pyplot.show()
print(torch.tensor(losses).mean())
print(torch.tensor(actual_losses).mean())
    
