################################
# Import Libraries
###############################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

# some settings
np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # We will use DataParallel, so the first GPU
    print(f"Running on GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
else:
    device = torch.device("cpu")
    print("Running on CPU")


################################
# FNO model
###############################
#Complex multiplication
def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer: FFT -> Linear Transform -> Inverse FFT  
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        # Perform Fourier transform
        batchsize = x.shape[0]
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        # Multiply top Fourier modes with Fourier weights
        out_ft = torch.zeros(batchsize, self.in_channels,  x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Perform Inverse Fourier transform
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=( x.size(-2), x.size(-1)))
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(SimpleBlock2d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        
        # Projection P
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # FNO Layer 1
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # FNO Layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # FNO Layer 3
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # FNO Layer 4
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # Projection Q
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        """
        A wrapper function
        """
        self.conv1 = SimpleBlock2d(modes1, modes2,  width)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c



################################################################
# configs
################################################################
################################################################
# configs
################################################################
TRAIN_PATH = 'train.mat'
TEST_PATH = 'test.mat'
VAL_PATH = 'val.mat'

ntrain = 8040
ntest = 2400
nval = 2400

batch_size = 40
learning_rate = 0.01

epochs = 500
step_size = 100
gamma = 0.5

modes1 = 10
modes2 = 40
width = 80


################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('X')
y_train = reader.read_field('Y')

s1, s2 = x_train.shape[1], x_train.shape[2]

reader.load_file(TEST_PATH)
x_test = reader.read_field('X')
y_test = reader.read_field('Y')

reader.load_file(VAL_PATH)
x_val = reader.read_field('X')
y_val = reader.read_field('Y')

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
x_val = x_normalizer.encode(x_val)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

grids = []
grids.append(np.linspace(0,3, s1))
grids.append(np.linspace(0,2, s2))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s1,s2,2)
grid = torch.tensor(grid, dtype=torch.float)

x_train = torch.cat([x_train.reshape(ntrain,s1,s2,1), grid.repeat(ntrain,1,1,1)], dim=3)
x_test = torch.cat([x_test.reshape(ntest,s1,s2,1), grid.repeat(ntest,1,1,1)], dim=3)
x_val = torch.cat([x_val.reshape(nval,s1,s2,1), grid.repeat(nval,1,1,1)], dim=3)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)


################################################################
# training and evaluation
################################################################
model = FNO2d(modes1, modes2, width).cuda()
print(model.count_params())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()

# Lists to store losses
train_losses = []
test_losses = []
val_losses = []

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            
    val_l2 = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            out = y_normalizer.decode(out)
            val_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    # Normalize the losses
    train_l2 /= ntrain
    test_l2 /= ntest
    val_l2 /= nval

    t2 = default_timer()
    print(f"Epoch: {ep}, Time Elapsed: {t2-t1}, Train Loss: {train_l2}, Validation Loss: {val_l2}, Test Loss: {test_l2}")
    
    # Store the losses for later plotting
    train_losses.append(train_l2)
    test_losses.append(test_l2)
    val_losses.append(val_l2)

################################################################
# Save model
################################################################
torch.save(model.state_dict(), 'trained_model.pth')
print("Model saved successfully as 'trained_model.pth'")

################################################################
# Plot random samples
################################################################

num_samples = 10
indices = random.sample(range(len(test_loader.dataset)), num_samples)

model.eval()  # Ensure model is in evaluation mode
for i, idx in enumerate(indices):
    # Get the sample from the dataset (assumed to return (x, y))
    x, y = test_loader.dataset[idx]
    # Add a batch dimension and move data to CUDA if available
    x = x.unsqueeze(0).cuda()
    y = y.unsqueeze(0).cuda()
    
    with torch.no_grad():
        pred = model(x)
        pred_decoded = y_normalizer.decode(pred)
    
    # Remove batch dimension and move to CPU
    x_sample = y.squeeze().cpu().numpy()
    y_sample = pred_decoded.squeeze().cpu().numpy()
    
    # Create a new figure with two subplots side by side
    plt.figure(figsize=(12, 6))
    
    # Plot the actual sample
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(x_sample, aspect='auto', cmap='jet', vmin=0, vmax=1)
    ax1.set_title(f"Actual_for_{np.unique(x_sample[0]).size - 1}_steps")
    ax1.invert_yaxis()  # Invert the y-axis
    plt.colorbar(im1, ax=ax1)
    
    # Plot the predicted sample
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(y_sample, aspect='auto', cmap='jet', vmin=0, vmax=1)
    ax2.set_title(f"Predicted_for_{np.unique(x_sample[0]).size - 1}_steps")
    ax2.invert_yaxis()  # Invert the y-axis
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    # Save the current figure as a high quality PDF file
    plt.savefig(f"plot_{i}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    
    # Optionally, display the plot
    plt.show()
    
    # Clear the figure after saving to free up memory
    plt.close()
