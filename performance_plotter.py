# %%
import pandas as pd
import pickle
from matplotlib import pyplot
import torch
from progress.bar import Bar

# Load the gpu (in my case it actually runs slower so i turned it off)
if torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cpu')
torch.no_grad()

    
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

f = open('nets/LSTM_ best_net_EURUSD_20.obj', 'rb')
net = pickle.load(f)
net.to(device)

batch_size = 20
real_open_prices = []
predicted_prices = []

net.eval()
epochs = len(norm_test_data) - 2*batch_size 
b = Bar('Generating...',max=epochs)
for epoch in range(epochs):
    test_index = batch_size + epoch
    for batch_num in range(batch_size):
        inpt = norm_test_data[test_index+batch_num]
        output = net(inpt)
    real_open_prices.append(unnormalize(float(norm_test_data[test_index+batch_size][0])))
    predicted_prices.append(unnormalize(float(output)))
    net.clean()
    b.next()


pyplot.plot(real_open_prices, label='Real Prices')
pyplot.plot(predicted_prices, label= 'Predicted Prices')
pyplot.legend()
pyplot.xlabel('Days')
pyplot.ylabel('EUR / USD')
pyplot.title('FGRNN 40 Predicitons')
pyplot.show()

    
