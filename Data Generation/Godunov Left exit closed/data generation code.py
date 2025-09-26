
# =============================================================================
# Import packages
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# Helper functions
# =============================================================================

def plot_initial_conditions(x_space, t_time, k_initial, q_entry, q_exit):

    fig, (ax1) = plt.subplots(1, 1, figsize=(4, 4))
    ax1.plot(x_space, k_initial)
    ax1.set_title("Initial Density", fontsize=14)
    ax1.set_xlabel("Space $x$ [m]", fontsize=12)
    ax1.set_ylabel(r"$\rho$ ($x,0$) [vehs/km]", fontsize=12)
    ax1.grid()

    fig.tight_layout()

    return fig, (ax1)



# =============================================================================
# Compute Turning Point- 2 methods
# =============================================================================

# def compute_psi(rho, x_grid, dx):

#     inv_1_minus_rho = 1 / (1 - rho)

#     # Cumulative sums from left and from right
#     cumsum_left = np.cumsum(inv_1_minus_rho) * dx
#     cumsum_right = np.cumsum(inv_1_minus_rho[::-1]) * dx
#     cumsum_right = cumsum_right[::-1]

#     # Find index where difference is minimal
#     min_diff_index = np.argmin(np.abs(cumsum_left - cumsum_right))
#     psi = x_grid[min_diff_index]
#     return psi


def compute_psi(rho, x_grid, dx):

    inv_1_minus_rho = 1.0 / (1.0 - rho)
    
    # Compute cumulative sum from the left (approximates the integral)
    cumsum = np.cumsum(inv_1_minus_rho) * dx
    total_area = cumsum[-1]
    half_area = total_area / 2.0
    
    # Find the first index where cumulative sum reaches or exceeds half_area
    index = np.searchsorted(cumsum, half_area)
    
    # Return the corresponding x value
    # If index is out of bounds, return the last grid point
    return x_grid[index] if index < len(x_grid) else x_grid[-1]


# =============================================================================
# Godunov Scheme
# =============================================================================


def fundamental_diag(k, k_max, v_max, k_cr, fd):
    return k*v_max*(1-k/k_max)


def Demandfn(k, k_max, v_max, k_cr, q_max, fd):
    if k <= k_cr:
        q = k*v_max*(1-k/k_max)
    else:
        q = q_max

    return q


def InvDemandfn_num(q, dem_fn, k_arr):
    qb = dem_fn[dem_fn < q][-1]
    qa = dem_fn[dem_fn >= q][0]
    kb = k_arr[dem_fn < q][-1]
    ka = k_arr[dem_fn >= q][0]
    k = kb + (ka-kb)*(q-qb)/(qa-qb)
    return k


def InvDemandfn(q, k_max, v_max, k_cr, q_max, v_free, fd):
    q = min(q, q_max)
    k = (k_max-np.sqrt(k_max**2-4*k_max/v_free*q))/2
    return k


def Supplyfn(k, k_max, v_max, k_cr, q_max, fd):
    if k >= k_cr:
        q = k*v_max*(1-k/k_max)
    else:
        q = q_max
    return q


def InvSupplyfn_num(q, sup_fn, k_arr):
    qb = sup_fn[sup_fn <= q][0]
    qa = sup_fn[sup_fn > q][-1]
    kb = k_arr[sup_fn <= q][0]
    ka = k_arr[sup_fn > q][-1]
    k = kb + (ka-kb)*(q-qb)/(qa-qb)
    return k


def InvSupplyfn(q, k_max, v_max, k_cr, q_max, v_free, fd):
    q = min(q, q_max)
    k = (k_max+np.sqrt(k_max**2-4*k_max/v_free*q))/2
    return k


def bound_cond_entry(k_prev, q_en, k_max, v_max, k_cr, q_max, v_free, fd):
    q_en = min(q_en, q_max)
    supply = Supplyfn(k_prev, k_max, v_max, k_cr, q_max, fd)
    if q_en <= supply:
        k = InvDemandfn(q_en, k_max, v_max, k_cr, q_max, v_free, fd)
    else:
        k = InvSupplyfn(q_en, k_max, v_max, k_cr, q_max, v_free, fd)
    return k


def bound_cond_exit(k_prev, q_ex, k_max, v_max, k_cr, q_max, v_free, fd):
    q_ex = min(q_ex, q_max)
    demand = Demandfn(k_prev, k_max, v_max, k_cr, q_max, fd)
    if q_ex < demand:
        k = InvSupplyfn(q_ex, k_max, v_max, k_cr, q_max, v_free, fd)
    else:
        k = InvDemandfn(q_ex, k_max, v_max, k_cr, q_max, v_free, fd)
    return k


def flux_function(k_xup, k_xdn, k_cr, q_max, k_max, v_max, fd):
    if (k_xdn <= k_cr) and (k_xup <= k_cr):
        q_star = fundamental_diag(k_xup, k_max, v_max, k_cr, fd)
    elif (k_xdn <= k_cr) and (k_xup > k_cr):
        q_star = q_max
    elif (k_xdn > k_cr) and (k_xup <= k_cr):
        q_star = min(fundamental_diag(k_xdn, k_max, v_max, k_cr, fd),
                     fundamental_diag(k_xup, k_max, v_max, k_cr, fd))
    elif (k_xdn > k_cr) and (k_xup > k_cr):
        q_star = fundamental_diag(k_xdn, k_max, v_max, k_cr, fd)
    return q_star


def density_update(k_x, k_xup, k_xdn, delt, delx, k_cr, q_max, k_max, v_max, fd):
    q_in = flux_function(k_xup, k_x, k_cr, q_max, k_max, v_max, fd)
    q_out = flux_function(k_x, k_xdn, k_cr, q_max, k_max, v_max, fd)
    k_x_nextt = k_x + (delt/delx)*(q_in - q_out)
    return k_x_nextt, q_out


def CFL_condition(delx, v_max):
    max_delt = delx/v_max
    return np.around(max_delt, 6)

# =================================================================================
# Solvers for left and right conservation laws
# =================================================================================

def solver_left(x_left, K_left, q_entry, q_exit, delt, delx, fd_type, t, k_jam, v_free):
                
    for x in reversed(x_left[0]):

     
        q_max = k_jam*v_free/4
        k_cr = k_jam/2

        # Get computational stencil
        k_x = K_left[t-1, x]

        # start @ 25
        if x == x_left[0][-1]:
            q_en = q_entry[t]
            k_xup = 0
        else:
            k_xup = K_left[t-1, x+1]

        # exit @ 0
        if x == x_left[0][0]:
            q_ex = q_exit[t]
            k_xdn = bound_cond_exit(
                    k_x, q_ex, k_jam, v_free, k_cr, q_max, v_free, fd_type)

        else:
            k_xdn = K_left[t-1, x-1]

        # Calculated and update new density
        k_x_next, q_out = density_update(k_x, k_xup, k_xdn, delt, delx, k_cr, q_max, k_jam, v_free, fd_type)
        K_left[t, x] = k_x_next
  
    return K_left


def solver_right(x_right, K_right, q_entry, q_exit, delt, delx, fd_type, t, k_jam, v_free):
    for x in range(len(x_right[0])):


        q_max = k_jam*v_free/4
        k_cr = k_jam/2

        # Get computational stencil
        # print(x, K_right.shape, x_right)
        k_x = K_right[t-1, x]

        # Start @ 25
        if x == 0:
            q_en = q_entry[t]
            k_xup = 0
        else:
            k_xup = K_right[t-1, x-1]

        # Exit @ 50
        if x == len(x_right[0])-1:
            q_ex = q_exit[t]
            k_xdn = 0
        else:
            k_xdn = K_right[t-1, x+1]

        # Calculated and update new density
        k_x_next, q_out = density_update(k_x, k_xup, k_xdn, delt, delx, k_cr, q_max, k_jam, v_free, fd_type)
        K_right[t, x] = k_x_next
      
    return K_right

# ===============================================================================
# Simulation steps - At TP randomly pick left or right as cost is same at each t
# ===============================================================================

def simulation(k_initial, q_entry, q_exit,
                   t_nums, x_nums, delt, delx, fd_params, k_jam_space):

    # FD parameters
    v_free = fd_params["v_free"]
    k_jam = fd_params["k_jam"]
    fd_type = fd_params["fd_type"]

    store_tp = []

    # Initialize time-space indices
    x_ind = np.arange(0, x_nums)
    t_ind = np.arange(0, t_nums)
    X_ind, T_ind = np.meshgrid(x_ind, t_ind)

    # Initialize K, Q matrix
    K = np.zeros((t_nums, x_nums))
    Q = np.zeros((t_nums, x_nums))

    K[0, :] = np.round(k_initial,2)    # at t = 0
    
    constant = 0
    turning_point = compute_psi(K[0, :],x_ind, delx, constant)



    if np.random.rand()  < 0.5:
        # With probability 1/2, the equality goes to the left group.
        x_left = np.where(x_ind <= turning_point)
        x_right = np.where(x_ind > turning_point)
    else:
        # With probability 1/2, the equality goes to the right group.
        x_left = np.where(x_ind < turning_point)
        x_right = np.where(x_ind >= turning_point)

    K_left = K[:, x_left[0]]
    K_right = K[:, x_right[0]]

    for t in range(1, X_ind.shape[0]):

        store_tp.append(turning_point)

        if(turning_point == x_ind[0]):
            K = solver_right([x_ind], K, q_entry, q_exit, delt, delx, fd_type, t, k_jam, v_free)
        elif(turning_point == x_ind[-1]):
            K = solver_left([x_ind], K, q_entry, q_exit, delt, delx, fd_type, t, k_jam, v_free)
        else:

            K_l = solver_left(x_left, K_left, q_entry, q_exit, delt, delx, fd_type, t, k_jam, v_free)
            K_r = solver_right(x_right, K_right, q_entry, q_exit, delt, delx, fd_type, t, k_jam, v_free)
    
            K  = np.hstack((K_l, K_r))

        finalvalue = 0  
        
        if t < 400:
            constant = 10000
            finalvalue = constant
        elif t < 800:
            constant = 0  # Corrected variable name
        else:
            constant = 10000

        turning_point = compute_psi(K[t, :],x_ind, delx, constant)

        if np.random.rand()  < 0.5:
            # With probability 1/2, the equality goes to the left group.
            x_left = np.where(x_ind <= turning_point)
            x_right = np.where(x_ind > turning_point)
        else:
            # With probability 1/2, the equality goes to the right group.
            x_left = np.where(x_ind < turning_point)
            x_right = np.where(x_ind >= turning_point)
    
        K_left = K[:, x_left[0]]
        K_right = K[:, x_right[0]]

        if (np.abs(K[t,:]) < 1e-3).all():
            return K, store_tp

    return K, store_tp


################################
# Generate initial conditions
###############################

rin = np.random.randint
run = np.random.uniform
rno = np.random.normal

def generate_rho(steps, target_std):
    # Initialize an array to hold the rho values.
    rho = np.zeros(steps + 1)
    
    # First value: uniformly chosen between 0.05 and 0.95, rounded to 2 decimals
    rho[0] = round(0.05 + 0.90 * np.random.rand(), 2)
    
    # Generate subsequent rho values based on the previous one
    for j in range(1, steps + 1):
        rho[j] = np.inf  # initialize with an invalid value
        # Keep generating until the new value meets the criteria
        while (rho[j] < 0.05 or rho[j] > 0.95 or abs(rho[j] - rho[j-1]) < 0.05):
            diff_rho = np.random.normal(0, target_std)
            new_value = rho[j-1] + diff_rho
            rho[j] = round(new_value, 2)
    
    return rho

def step_func_gen(x_grid, kmax, num_steps, num_points, step_height_std):
    # Continuously generate until the profile has exactly num_steps+1 unique values after rounding.
    while True:
        # Randomly select num_steps unique step positions (not including the start or end)
        step_positions = np.sort(np.random.choice(range(int(num_points/50), num_points- int(num_points/50)), size=num_steps, replace=False))
        # Include the start and end of the grid as boundaries
        boundaries = np.concatenate(([0], step_positions, [num_points]))
        
        # Generate density values for each segment.
        # Start with an initial density chosen uniformly from 0.01 to 0.99.
        densities = generate_rho(num_steps, step_height_std)
        
        # Now assign each density value to the corresponding region defined by the boundaries.
        k_initial = np.empty(num_points, dtype=float)
        for i in range(len(densities)):
            start = boundaries[i]
            end = boundaries[i+1]
            k_initial[start:end] = densities[i]
        
        # Round values to two decimals.
        k_initial_rounded = np.round(k_initial, 2)
        
        # Verify that we have exactly num_steps+1 unique values after rounding.
        if len(np.unique(k_initial_rounded)) == num_steps + 1:
            return k_initial_rounded.astype('float')

################################
# Generate dataset
###############################
def create_train_data(K):
    # Create a deep copy of the list so as not to modify the original arrays.
    train_x = [np.copy(arr) for arr in K]
    # For every array except the first, set all elements to zero.
    for i in range(1, len(train_x)):
        train_x[i] = np.zeros_like(train_x[i])
    return train_x

    # ------------------------------
# ------------------------------
# User-specified parameters:
step_options = [1,2,3]  # List of num_steps values to generate (can be any integers)
n_train = 1000             # Number of training samples per step group
n_val   = 300        # Number of validation samples per step group
n_test  = 300        # Number of test samples per step group
total_group = n_train + n_val + n_test  # Total samples for each step group
# max_time_lim = 1400
# min_time_value = 650
downsample_factor_t = 36
downsample_factor_x = 5
tps_max = 400
step_height_std = 0.5
k_max = 1
v_max = 1

fd_params = {
    "k_jam": k_max,
    "v_free": v_max,
    "fd_type": "Greenshield"
}


x_max = 2                  # road length in kilometres
t_max = 3                   # time period of simulation in hours
delx = 1/500                      # cell length in kilometres
delt = 1/600                       # time discretization in hours
x_nums = round(x_max/delx)
t_nums = round(t_max/delt)

q_entry = np.random.uniform(0,0, t_nums)
q_exit = np.random.uniform(0,0, t_nums)
k_jam_space = np.repeat(fd_params["k_jam"], x_nums)

x_ind = np.arange(0, x_nums)


# ------------------------------
# Initialize a dictionary to hold only Y datasets for each step group and split.
datasets = {}
for steps in step_options:
    datasets[steps] = {
        'train': {'X': [], 'Y': []},
        'val':   {'X': [], 'Y': []},
        'test':  {'X': [], 'Y': []}
    }

unique_k_samples = set()

# ------------------------------
# Data Generation Loop:
# For each step option, generate the required number of samples.
for steps in step_options:
    print(f"Generating samples for {steps}-step data...")
    for local_index in range(total_group):

        print(local_index)
        accepted_sample = False
        # Continue trying until an acceptable sample is generated.
        while not accepted_sample:
            k_initial = step_func_gen(x_ind, k_max, steps, x_nums, step_height_std)

            rounded_k = tuple(np.round(np.unique(k_initial), 2))
            # If the sample is not unique, skip to the next iteration.
            if rounded_k in unique_k_samples:
                continue
            # Otherwise, add the new unique sample to the set.
            unique_k_samples.add(rounded_k)

            K, tps = simulation(k_initial, q_entry, q_exit,
                               t_nums, x_nums, delt, delx, fd_params, k_jam_space)

            
            # if (np.max(tps) - np.min(tps)) > tps_max:
            #     print(f"Step {steps} sample {local_index} discarded due to tps range > {tps_max}")
            #     continue
                
            # # Check if the number of time steps is within the desired range.
            # if len(K) < min_time_value or len(K) > max_time_lim:
            #     print(f"Step {steps} sample {local_index} discarded due to time steps length {len(K)} outside [{min_time_value},{max_time_lim}]")
            #     continue

            # if not np.any(K[ :150,:] < 0.01):
            #     print(f"Step {steps} sample {local_index} discarded due to no K value < 0.01 for t < 120")
            #     continue

            

  
            # Check if any array in K has values outside the range [0, 1]
            if any(np.any(arr < 0) or np.any(arr > 1) for arr in K):
                print(f"Step {steps} sample {local_index} discarded due to K values outside [0, 1]")
                continue
            accepted_sample = True

        # Deep copy of K for storage.
        K_copy = [np.copy(arr) for arr in K]
        # Create a train version of K (first time step preserved, others zeroed).
        train_x_data = create_train_data(K)
        
        # Assign the sample based on local_index.
        if local_index < n_train:
            datasets[steps]['train']['Y'].append(K_copy)
            datasets[steps]['train']['X'].append(train_x_data)
        elif local_index < n_train + n_val:
            datasets[steps]['val']['Y'].append(K_copy)
            datasets[steps]['val']['X'].append(train_x_data)
        else:
            datasets[steps]['test']['Y'].append(K_copy)
            datasets[steps]['test']['X'].append(train_x_data)

# ------------------------------
# Padding: Ensure all samples have the same number of time steps.
def pad_sample(sample, max_steps):
    padded_sample = sample.copy()
    current_steps = len(padded_sample)
    if current_steps < max_steps:
        pad_shape = padded_sample[0].shape  # assume same shape per time step
        for _ in range(max_steps - current_steps):
            padded_sample.append(np.zeros(pad_shape))
    return padded_sample

def pad_and_stack(dataset, max_steps):
    new_dataset = []
    for sample in dataset:
        padded_sample = pad_sample(sample, max_steps)
        sample_array = np.stack(padded_sample, axis=0)
        new_dataset.append(sample_array)
    return new_dataset

# Determine maximum number of time steps across all datasets.
all_samples = []
for steps in step_options:
    for split in ['train', 'val', 'test']:
        all_samples.extend(datasets[steps][split]['Y'])
        all_samples.extend(datasets[steps][split]['X'])
if all_samples:
    max_steps = max(len(sample) for sample in all_samples)
else:
    raise ValueError("No samples generated!")
print("Max time steps across all samples:", max_steps)

# Pad and stack every dataset.
for steps in step_options:
    for split in ['train', 'val', 'test']:
        datasets[steps][split]['Y'] = pad_and_stack(datasets[steps][split]['Y'], max_steps)
        datasets[steps][split]['X'] = pad_and_stack(datasets[steps][split]['X'], max_steps)

# ------------------------------
# Downsampling: For example, take every n-th time step.
def downsample_dataset(dataset, n, m):
    return [sample[::n, ::m] for sample in dataset]


for steps in step_options:
    for split in ['train', 'val', 'test']:
        datasets[steps][split]['Y'] = downsample_dataset(datasets[steps][split]['Y'], downsample_factor_t, downsample_factor_x)
        datasets[steps][split]['X'] = downsample_dataset(datasets[steps][split]['X'], downsample_factor_t, downsample_factor_x)



################################
# Plot initial conditions
###############################
import matplotlib.pyplot as plt

# Select one sample from each step group from the training set.
sample1 = datasets[1]['train']['Y'][1]
sample2 = datasets[2]['train']['Y'][1]
sample3 = datasets[3]['train']['Y'][1]

# Create a figure with three subplots.
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot the first feature (column 0) of each sample.
axs[0].plot(sample1[0, :])
axs[0].set_title("1-Step Sample (TrainY) - Feature 0")
axs[0].set_xlabel("Time Step")
axs[0].set_ylabel("Value")

axs[1].plot(sample2[0, :])
axs[1].set_title("2-Step Sample (TrainY) - Feature 0")
axs[1].set_xlabel("Time Step")
axs[1].set_ylabel("Value")

axs[2].plot(sample3[0, :])
axs[2].set_title("3-Step Sample (TrainY) - Feature 0")
axs[2].set_xlabel("Time Step")
axs[2].set_ylabel("Value")

plt.tight_layout()
plt.show()


################################
# Plot samples
###############################

import matplotlib.pyplot as plt
import random

# Select one random sample from each step group training set.
random_sample1 = random.choice(datasets[1]['train']['Y'])
random_sample2 = random.choice(datasets[2]['train']['Y'])
random_sample3 = random.choice(datasets[3]['train']['Y'])

# Create a figure with three subplots (one per step group).
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

vmin=0
vmax=1

# Plot 1-Step sample.
im1 = axs[0].imshow(random_sample1, aspect='auto', origin='lower',  cmap='jet', vmin=vmin, vmax=vmax)
axs[0].set_title("Random 1-Step Sample (TrainY)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("Time")
fig.colorbar(im1, ax=axs[0])

# Plot 2-Step sample.
im2 = axs[1].imshow(random_sample2, aspect='auto', origin='lower',  cmap='jet',vmin=vmin, vmax=vmax)
axs[1].set_title("Random 2-Step Sample (TrainY)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("Time")
fig.colorbar(im2, ax=axs[1])

# Plot 3-Step sample.
im3 = axs[2].imshow(random_sample3, aspect='auto', origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
axs[2].set_title("Random 3-Step Sample (TrainY)")
axs[2].set_xlabel("x")
axs[2].set_ylabel("Time")
fig.colorbar(im3, ax=axs[2])

plt.tight_layout()
plt.show()

################################
# Save Datasets
###############################
import numpy as np
from scipy.io import savemat

# Use integer keys based on your datasets dictionary
step_options = [1, 2, 3]
splits = ['train', 'val', 'test']
base_path = '//datasets//'

for split in splits:
    combined_X = []
    combined_Y = []
    for step in step_options:
        # Directly access the data from your datasets dictionary using integer keys
        data = datasets[step][split]
        combined_X.append(data['X'])
        combined_Y.append(data['Y'])
    
    # Concatenate arrays along the first axis (adjust axis if needed)
    combined_X = np.concatenate(combined_X, axis=0)
    combined_Y = np.concatenate(combined_Y, axis=0)
    
    # Create a dictionary for the combined data
    combined_data = {'X': combined_X, 'Y': combined_Y}
    
    # Save the combined data as a .mat file
    mat_filename = f'{base_path}combined_{split}_easier.mat'
    savemat(mat_filename, combined_data)
    print(f"Saved combined {split} data to {mat_filename}")
