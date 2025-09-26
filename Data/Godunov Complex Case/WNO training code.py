################################################################
# Import libraries
################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution_v3 import WaveConv2dCwt
import pickle as pkl

torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # We will use DataParallel, so the first GPU
    print(f"Running on GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
else:
    device = torch.device("cpu")
    print("Running on CPU")

################################################################
# Model
################################################################


""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, omega, padding=0):
        super(WNO2d, self).__init__()


        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet1 = wavelet[0]
        self.wavelet2 = wavelet[1]
        self.omega = omega
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2dCwt(self.width, self.width, self.level, size=self.size,
                                            wavelet1=self.wavelet1, wavelet2=self.wavelet2, omega=self.omega) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
#         print('gird:',grid.shape)
        x = torch.cat((x, grid), dim=-1)  
#         print('x:', x.shape)
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)
#         print('x after permute', x.shape)# Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
#         print('after padding:', x.shape) # (129,129) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
#             print(convl(x).shape, wl(x).shape, x.shape)
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.mish( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# Model configurations
################################################################

ntrain = 8040
ntest = 2400
nval = 2400

batch_size = 40
learning_rate = 0.01

epochs = 500
step_size = 50   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = ['near_sym_b', 'qshift_b']  # wavelet basis function
level = 2        # lavel of wavelet decomposition
width = 80       # uplifting dimension
layers = 4       # no of wavelet layers

# sub = 5
# h = int(((421 - 1)/sub) + 1) # total grid size divided by the subsampling rate
grid_range = [2, 3]          # The grid boundary in x and y direction
in_channel = 3   # (a(x, y), x, y) for this case

################################################################
# Load datasets
################################################################
TRAIN_PATH = 'train.mat'
TEST_PATH = 'test.mat'
VAL_PATH = 'val.mat'

################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('X')[:ntrain,:-1,1:]
y_train = reader.read_field('Y')[:ntrain,:-1,1:]

s1, s2 = x_train.shape[1], x_train.shape[2]

reader.load_file(TEST_PATH)
x_test = reader.read_field('X')[:ntest,:-1,1:]
y_test = reader.read_field('Y')[:ntest,:-1,1:]

reader.load_file(VAL_PATH)
x_val = reader.read_field('X')[:nval,:-1,1:]
y_val = reader.read_field('Y')[:nval,:-1,1:]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
x_val = x_normalizer.encode(x_val)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s1,s2,1)
x_test = x_test.reshape(ntest,s1,s2,1)
x_val = x_val.reshape(nval,s1,s2,1)


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

################################################################
# The model definition
################################################################
model = WNO2d(width=width, level=level, layers=layers, size=[s1,s2], wavelet=wavelet,
              in_channel=in_channel, grid_range=grid_range, omega=8, padding=1).to(device)
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
val_loss = torch.zeros(epochs)

myloss = LpLoss(size_average=False)
y_normalizer.to(device)


for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        
        out = model(x).reshape(batch_size, s1, s2)
#         print(out.shape)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()
        optimizer.step()
        
        train_mse += mse.item()
        train_l2 += loss.item()
    
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x).reshape(batch_size, s1, s2)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()
            
            
    val_l2 = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x).reshape(batch_size, s1, s2)
            out = y_normalizer.decode(out)

            val_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_mse /= len(train_loader)
    train_l2/= ntrain
    test_l2 /= ntest
    val_l2 /= nval
    
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2
    val_loss[ep] = val_l2
    
    t2 = default_timer()
    print("Epoch-{}, Time-{:0.4f}, Train-MSE-{:0.4f}, Train-L2-{:0.4f}, Test-L2-{:0.4f}, val-L2-{:0.4f}"
          .format(ep, t2-t1, train_mse, train_l2, test_l2, val_l2))