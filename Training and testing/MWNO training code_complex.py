################################################################
# Import libraries
################################################################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Tuple
import math

from models import *
from utilities3 import *
from utils import *
import pickle as pkl

from functools import partial

from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt


from scipy.io import loadmat, savemat
import math
import os
import h5py

from timeit import default_timer
torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # We will use DataParallel, so the first GPU
    print(f"Running on GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
else:
    device = torch.device("cpu")
    print("Running on CPU")

def get_initializer(name):
    
    if name == 'xavier_normal':
        init_ = partial(nn.init.xavier_normal_)
    elif name == 'kaiming_uniform':
        init_ = partial(nn.init.kaiming_uniform_)
    elif name == 'kaiming_normal':
        init_ = partial(nn.init.kaiming_normal_)
    return init_

################################################################
# Configs
################################################################
ntrain = 8040
ntest = 2400
nval = 2400

batch_size = 50

r = 1
h = int(((512 - 1)/r) + 1)
s = h

TRAIN_PATH = 'train.mat'
TEST_PATH = 'test.mat'
VAL_PATH = 'val.mat'


################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('X')[:ntrain,:,:]
y_train = reader.read_field('Y')[:ntrain,:,:]

s1, s2 = x_train.shape[1], x_train.shape[2]

reader.load_file(TEST_PATH)
x_test = reader.read_field('X')[:ntest,:,:]
y_test = reader.read_field('Y')[:ntest,:,:]

reader.load_file(VAL_PATH)
x_val = reader.read_field('X')[:nval,:,:]
y_val = reader.read_field('Y')[:nval,:,:]


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
# Model definition
################################################################
ich = 3
initializer = get_initializer('xavier_normal') # xavier_normal, kaiming_normal, kaiming_uniform

torch.manual_seed(0)
np.random.seed(0)

model = MWT2d(ich, 
            alpha = 12,
            c = 4,
            k = 4, 
            base = 'legendre', # 'chebyshev'
            nCZ = 4,
            L = 0,
            initializer = initializer,
            ).to(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

learning_rate = 0.01

epochs = 500
step_size = 100
gamma = 0.5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
import time
myloss = LpLoss(size_average=False)
y_normalizer.cuda()

for epoch in range(1, epochs+1):
    start_time = time.time()
    train_l2 = train(model, train_loader, optimizer, epoch, device,
        lossFn = myloss, lr_schedule = scheduler,
        post_proc = y_normalizer.decode)
    epoch_time = time.time() - start_time
    val_l2 = test(model, val_loader, device, lossFn=myloss, post_proc=y_normalizer.decode)
    test_l2 = test(model, test_loader, device, lossFn=myloss, post_proc=y_normalizer.decode)
    
    print(f'epoch: {epoch}, time ={epoch_time:.3f}, train l2 = {train_l2}, val l2 = {val_l2}, test l2 = {test_l2}')

################################################################
# save model
################################################################
torch.save(model.state_dict(), 'trained_model_mwno.pth')
print("Model saved successfully as 'trained_model.pth'")
