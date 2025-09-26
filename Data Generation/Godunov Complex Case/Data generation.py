
# =============================================================================
# Import packages
# =============================================================================

import os
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
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

        # start 
        if x == x_left[0][-1]:
            q_en = q_entry[t]
            k_xup = 0
        else:
            k_xup = K_left[t-1, x+1]

        # exit 
        if x == x_left[0][0]:
            q_ex = q_exit[t]
            k_xdn = 0

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

        # Start 
        if x == 0:
            q_en = q_entry[t]
            k_xup = 0
        else:
            k_xup = K_right[t-1, x-1]

        # Exit 
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

    turning_point = compute_psi(K[0, :],x_ind, delx)

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

        turning_point = compute_psi(K[t, :],x_ind, delx)

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

def downsample_array(arr, target_shape=(50, 200)):

    orig_shape = arr.shape
    # Compute uniformly spaced indices for both dimensions.
    t_indices = np.linspace(0, orig_shape[0] - 1, target_shape[0], dtype=int)
    x_indices = np.linspace(0, orig_shape[1] - 1, target_shape[1], dtype=int)
    
    # Ensure the first element (row 0) is always included for the X array.
    # np.linspace with a start of 0 ensures that t_indices[0] == 0.
    if t_indices[0] != 0:
        t_indices[0] = 0
        
    return arr[np.ix_(t_indices, x_indices)]


# ------------------------------
# User-specified parameters:
step_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # List of num_steps values to generate (can be any integers)
n_train = 850             # Number of training samples per step group
n_val   =  250     # Number of validation samples per step group
n_test  = 250        # Number of test samples per step group
total_group = n_train + n_val + n_test  # Total samples for each step group
# max_time_lim = 1400
# min_time_value = 650
downsample_factor_t = 14
downsample_factor_x = 5
tps_max = 500
tps_min = 150
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
base_dir = "data_samples"
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "val")
test_dir  = os.path.join(base_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

unique_k_samples = set()

# ------------------------------
# Data Generation Loop:
# For each step option, generate the required number of samples.
for steps in step_options:
    print(f"Generating samples for {steps}-step data...")
    for local_index in range(total_group):
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

            
            diff = np.max(tps) - np.min(tps)
            if diff > tps_max or diff < tps_min:
                print(f"Step {steps} sample {local_index} discarded due to tps range {diff} not in range [{tps_min}, {tps_max}]")
                continue
                
            # # Check if the number of time steps is within the desired range.
            # if len(K) < min_time_value or len(K) > max_time_lim:
            #     print(f"Step {steps} sample {local_index} discarded due to time steps length {len(K)} outside [{min_time_value},{max_time_lim}]")
            #     continue

            if not np.any(K[ :150,:] < 0.01):
                print(f"Step {steps} sample {local_index} discarded due to no K value < 0.01 for t < 120")
                continue


            # Check if any array in K has values outside the range [0, 1]
            if any(np.any(arr < 0) or np.any(arr > 1) for arr in K):
                print(f"Step {steps} sample {local_index} discarded due to K values outside [0, 1]")
                continue
            accepted_sample = True


        train_x_data = create_train_data(K)

        # Determine the split based on local_index.
        if local_index < n_train:
            out_dir = train_dir
        elif local_index < n_train + n_val:
            out_dir = val_dir
        else:
            out_dir = test_dir
        
        # Build filename as "filenumber_numberOfSteps.npz" e.g. "0001_1.npz"
        filename = f"{local_index:04d}_{steps}.npz"
        filepath = os.path.join(out_dir, filename)
        
        # Save both X and Y in a single file.
        train_x_down = downsample_array(train_x_data, target_shape=(50, 200))
        K_down = downsample_array(K, target_shape=(50, 200))
        
        np.savez(filepath, X=train_x_down, Y=K_down)

################################
# Downsample files to 50,200 if required
###############################

def downsample_array(arr, target_shape=(50, 200)):

    orig_shape = arr.shape
    # Compute uniformly spaced indices for both dimensions.
    t_indices = np.linspace(0, orig_shape[0] - 1, target_shape[0], dtype=int)
    x_indices = np.linspace(0, orig_shape[1] - 1, target_shape[1], dtype=int)
    
    # Ensure the first element (row 0) is always included for the X array.
    # np.linspace with a start of 0 ensures that t_indices[0] == 0.
    if t_indices[0] != 0:
        t_indices[0] = 0
        
    return arr[np.ix_(t_indices, x_indices)]

folders = ['train', 'test', 'val']

for folder in folders:
    print(f"Processing folder: {folder}")
    files = [f for f in os.listdir(folder) if f.endswith('.npz')]
    print(f"Found {len(files)} files in {folder}.")
    
    for file in files:
        file_path = os.path.join(folder, file)
        
        # Load the file which contains two arrays: X and Y.
        with np.load(file_path) as data:
            X = data['X']
            Y = data['Y']
        
 
        X_new = downsample_array(X, target_shape=(50, 200))
        Y_new = downsample_array(Y, target_shape=(50, 200))
        
        # Overwrite the original file with the new downsampled arrays.
        np.savez(file_path, X=X_new, Y=Y_new)
        print(f"Processed file: {file}")




################################
# Create train, test and val .mat files
###############################

def combine_npz_to_mat(folder, output_file):
    """
    Combines all .npz files in the specified folder into one .mat file.
    Each .npz file is assumed to contain arrays with keys 'X' and 'Y', each of shape (128,128).
    
    The combined data will be stored as 3D arrays of shape (N, 128, 128) for both X and Y,
    where N is the number of files. The output .mat file is compressed to minimize file size.
    """
    files = [f for f in os.listdir(folder) if f.endswith('.npz')]
    combined_X = []
    combined_Y = []
    
    for idx, file in enumerate(files):
        file_path = os.path.join(folder, file)
        with np.load(file_path) as data:
            # Load the arrays under keys 'X' and 'Y'
            X = data['X']
            Y = data['Y']
        combined_X.append(X)
        combined_Y.append(Y)
        
        if (idx + 1) % 100 == 0:
            print(f"{folder}: Processed {idx + 1} files")
    
    # Stack the arrays along a new axis (axis=0) so that the resulting shape is (N, 128, 128)
    combined_X = np.stack(combined_X, axis=0)
    combined_Y = np.stack(combined_Y, axis=0)
    
    # Save the combined arrays to a .mat file with compression enabled
    savemat(output_file, {'X': combined_X, 'Y': combined_Y}, do_compression=True)
    print(f"Saved combined file: {output_file}")

folders = ['train', 'test', 'val']
for folder in folders:
    output_file = f"{folder}.mat"
    combine_npz_to_mat(folder, output_file)


################################
# Plot a few samples
###############################

# List of folders (which correspond to our mat files)
folders = ['train', 'test', 'val']
# Indices of samples to plot (here: first three samples)
sample_indices = [0, 1, 2]

for folder in folders:
    mat_file = f"{folder}.mat"
    data = loadmat(mat_file)
    
    # Assume data is stored under keys 'X' and 'Y'
    X_data = data['X']
    Y_data = data['Y']
    num_samples = X_data.shape[0]
    print(f"{folder}: {num_samples} samples loaded from {mat_file}.")
    
    # Create a figure: one row per sample, two columns (X and Y)
    n_samples_to_plot = min(len(sample_indices), num_samples)
    fig, axes = plt.subplots(n_samples_to_plot, 2, figsize=(10, 3 * n_samples_to_plot))
    
    # If there's only one sample, axes may not be 2D; make it 2D for consistent indexing
    if n_samples_to_plot == 1:
        axes = np.array([axes])
    
    for i, idx in enumerate(sample_indices):
        if idx >= num_samples:
            break  # Avoid index error if fewer samples than expected
        # Plot X as a heatmap
        ax1 = axes[i, 0]
        im1 = ax1.imshow(X_data[idx], aspect='auto', cmap='jet')
        ax1.set_title(f'{folder} Sample {idx} - X')
        fig.colorbar(im1, ax=ax1)
        
        # Plot Y as a heatmap
        ax2 = axes[i, 1]
        im2 = ax2.imshow(Y_data[idx], aspect='auto', cmap='jet')
        ax2.set_title(f'{folder} Sample {idx} - Y')
        fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()



