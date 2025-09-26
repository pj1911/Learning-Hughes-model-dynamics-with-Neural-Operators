
# =============================================================================
# Import packages
# =============================================================================
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re
import torch
import seaborn as sns
import pickle as pkl
from scipy.io import savemat

folder_path = '//WFTalgo final code//complex//train//'  # similar for train, val and test

# =============================================================================
# Insights on data
# =============================================================================

def read_all_mat_files_info(folder_path):
    mat_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]
    longest_time = []
    longest_x = []
    
    # Read each file
    for mat_file in mat_files:

        mat_data = scipy.io.loadmat(mat_file)
        matrix = mat_data['all_stored_data']
        
        longest_time.append(matrix.shape[0])
        
      
        for i in range(matrix.shape[0]):
            longest_x.append(len(matrix[i][0]))

    return longest_time, longest_x
longest_time,longest_x = read_all_mat_files_info(folder_path)
# Plot histogram
plt.hist(longest_time, bins=20, edgecolor='black')
plt.title('Histogram of time length of samples')
plt.xlabel('number of time steps')
plt.ylabel('Number of samples')
plt.show()

# Plot histogram
plt.hist(longest_x, bins=10, edgecolor='black')
plt.title('Histogram of most discretization of x')
plt.xlabel('number of discretizations')
plt.ylabel('Frequency')
# plt.ylim(0,100)
plt.show()

# =============================================================================
# Find Outliers
# =============================================================================

def read_all_mat_files(folder_path):
    mat_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]
    longest_time = []
    longest_x = []
    
    # Read each file
    for mat_file in mat_files:

        mat_data = scipy.io.loadmat(mat_file)
        matrix = mat_data['all_stored_data']
        if(matrix.shape[0]>650):
            longest_time.append(matrix.shape[0]) # 600,2 where t = 600 and col1 = x_val and col2 = rho_val
            os.remove(mat_file)

        max_x_len = []
        for i in range(matrix.shape[0]):
            max_x_len.append(len(matrix[i][0]))
        
        max_val = max(max_x_len)
        if max_val>201:
            longest_x.append(max_val)


    return longest_time, longest_x
outliers_time,outliers_x = read_all_mat_files(folder_path)

# Plot histogram
plt.hist(outliers_time, bins=30, edgecolor='black')
plt.title('Histogram of total steps of samples')
plt.xlabel('total steps')
plt.ylabel('Number of samples')
plt.show()

# Plot histogram
plt.hist(outliers_x, bins=30, edgecolor='black')
plt.title('Histogram of len(x) of samples')
plt.xlabel('len(x) ')
plt.ylabel('Number of samples')
plt.show()

# =============================================================================
# Get list of outliers
# =============================================================================

def get_outliers_name(folder_path):
    mat_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]
    longest_x = []

    list_of_outliers = []

    # Read each file
    for mat_file in mat_files:
        mat_data = scipy.io.loadmat(mat_file)
        matrix = mat_data['all_stored_data']

        total_time_till_now = 0

        for row in matrix:

            t = row[2].reshape(row[2].shape[1])  # Adjust indexing based on your matrix structure
   
            


            # Check for outliers in row[0] or row[1]
            col_0 = row[0].reshape(-1) if hasattr(row[0], 'shape') else row[0]
            col_1 = row[1].reshape(-1) if hasattr(row[1], 'shape') else row[1]
            col_2 = row[2].reshape(-1) if hasattr(row[2], 'shape') else row[2]

    


            if any( (col_0 < -1) | (col_0 > 1) ) or any( (col_1 < 0) | (col_1 > 1) ):
                print(f"Outlier detected in file: {mat_file}")
                print(f"  Matrix time count: {matrix.shape[0]}\n")
                list_of_outliers.append(mat_file)
                os.remove(mat_file)
                break

            if( (col_2[0] < -1) | (col_2[0] > 1) ): 
                print(f"Outlier detected in turning point: {mat_file}")
                print(f"  Matrix time count: {matrix.shape[0]}\n")
                list_of_outliers.append(mat_file)
                os.remove(mat_file)
                break

        
        # Check for time error if matrix rows exceed 1000
        if matrix.shape[0] > 650:
            print(f"Time error in file: {mat_file}")
            print(f"  Matrix row count: {matrix.shape[0]}\n")
            list_of_outliers.append(mat_file)
            os.remove(mat_file)


            # Check for time error if matrix rows exceed 1000
        if matrix.shape[0] < 100:
            print(f"Time error in file: {mat_file}")
            print(f"  Matrix row count: {matrix.shape[0]}\n")
            list_of_outliers.append(mat_file)
            os.remove(mat_file)

 

    return list_of_outliers
list_of_outliers = get_outliers_name(folder_path)


# =============================================================================
# Make all samples of same size
# =============================================================================

def read_all_mat_files_for_train(folder_path, list_of_outliers):
    # Normalize the outlier file paths
    normalized_outliers = {os.path.normpath(f) for f in list_of_outliers}
    
    # List to store names of files added to all_data
    dataset_names = []
    
    # List all .mat files in the folder
    mat_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]

    all_data = []  # List to store 3D data from all files
    processed_count = 0

    stored_values = []

    total_time_till_now = 0

    # Read each file, ignoring the ones in the outliers list
    for mat_file in mat_files:
        # Normalize the current file path
        normalized_mat_file = os.path.normpath(mat_file)
        
        # # Skip files that are in the outliers list
        if normalized_mat_file in normalized_outliers:
            print(f"Skipping outlier file: {normalized_mat_file}")
            continue

        # Process the file if it's not an outlier
        mat_data = scipy.io.loadmat(mat_file)
        matrix = mat_data['all_stored_data']

        dataset_names.append(mat_file)

       
        # print(mat_file,processed_count)
        processed_count += 1

              
        interpolated_results = []
        for i, row in enumerate(matrix): 
  
            rho = row[1].reshape(row[1].shape[0])
            turning_pt = np.round(row[2].reshape(row[2].shape[0])[0], 2)

            # if i == 0:
            #     rho = rho[:-1]  # Remove the last element from `rho`
            # rho = np.insert(rho, 0, 0)  # Insert 0 at the start
            # rho = np.append(rho, 0)     # Append 0 at the end
           
            rho = np.append(rho, turning_pt)

            interpolated_results.append(rho)

        # Convert the list to a numpy array
        interpolated_results = np.array(interpolated_results)
        current_rows = interpolated_results.shape[0]
        total_time_till_now = ( current_rows / 650 ) * 3
        if current_rows < 650:
        
            # zero_padding = np.ones((350 - current_rows, xmax))     # ones padding
            zero_padding = np.zeros((650 - current_rows, 202))
            interpolated_results = np.vstack([interpolated_results, zero_padding])
            # twos_padding = np.full((300 - current_rows, 202), 0.1)
            # interpolated_results = np.vstack([interpolated_results, twos_padding])
            
        # Add total_time_till_now as a new column after padding
        time_column = np.full((interpolated_results.shape[0], 1), total_time_till_now)
        interpolated_results = np.hstack([interpolated_results, time_column])
        # print(interpolated_results.shape)
        all_data.append(interpolated_results)

    all_data = np.array(all_data)
    count_greater_than_30 = len(stored_values)
    print(count_greater_than_30)
    
    return dataset_names, all_data

data_names_with_index, our_new_dataset = read_all_mat_files_for_train(folder_path, list_of_outliers)

max_len = 0
# Loop through each element in the dataset
for item in our_new_dataset:
    # Check the length along the first dimension
    item_len = item.shape[0]
    # Update max_len if the current item length is greater
    if item_len > max_len:
        max_len = item_len
print("Maximum length found:", max_len)

def remove_nans(data):
    nan_mask = np.any(np.isnan(data), axis=(1, 2))
    filtered_dataset = data[~nan_mask]
    return filtered_dataset
our_new_dataset = remove_nans(our_new_dataset)
our_new_dataset.shape

# =============================================================================
# Get training data
# =============================================================================

# Function to process the data into input (X_train) and output (y_train)
def process_data_for_training(data, k_jam):
    TrainX = []
    TrainY = []
    turning_pointsX = []
    turning_pointsY = []
    total_time =[]
    
    # Iterate over each file's data
    for k in data:

        first_2 = k[:2, :]
        # Every 5th time step after the first 10
        every_5th = k[2::13, :]
        # Combine the two parts
        result = np.vstack((first_2, every_5th))
        tt_out = result[:,-1].copy().astype('float32')
        tt_out_expanded = np.tile(tt_out[:, np.newaxis], (1, 201)) 
        total_time.append(tt_out_expanded)
        K_out = result[:,:-2].copy().astype('float32')
        # Create input data (X) and modify according to your rules
        K_inp = result[:,:-2].copy().astype('float32')
        print(K_inp.shape)
        # K_inp[1:, 1:-1] = k_jam # save the boundary values
        K_inp[1:, :] = k_jam
        # Store the modified input and output
        TrainX.append(K_inp)
        TrainY.append(K_out)  # Ensure TrainY is the output
    
   
    
    return TrainX, TrainY, total_time
TrainX, TrainY, total_time_train = process_data_for_training(our_new_dataset,0)



# =============================================================================
# Save training data
# =============================================================================
data = {'X': TrainX, 'Y': TrainY}
savemat('//WFTalgo final code//complex2//traincomplex.mat', data)

# =============================================================================
# Plot a few random samples
# =============================================================================

# Move the tensor to the CPU before converting it to a numpy array
random_k_values = np.random.randint(0, 1300, 40)
# Iterate over the random k values and plot the heatmap with turning points
for k in (random_k_values):
    
    tensor_np = TrainY[k]
    # current_turning_points = turning_pointsY_train[k][:,0]*100 + 100  # Assuming this is a 1D array of length 300
    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(tensor_np, cmap="viridis", cbar=True, vmin=0, vmax=1)
    
    plt.gca().invert_yaxis()  # Invert the y-axis for consistency with the heatmap
    plt.xlabel("X-axis")
    plt.ylabel("Time (y-axis)")
    # plt.legend()
    plt.title("Heatmap with Turning Points Overlay")
    plt.show()
