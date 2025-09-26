%% Disable stopping on errors and clear all breakpoints
dbclear all;
dbclear if error;


%% Set parameters
nSteps = 10;          % Maximum number of steps (jumps) to simulate; adjust as needed
nTrain = 825;       % Number of training samples per step
nTest  = 250;        % Number of testing samples per step
nVal   = 250;        % Number of validation samples per step
total_samples = nTrain + nTest + nVal;
target_std = 0.5;    % Target standard deviation for the rho difference
epsilon = 1/200;     % Given epsilon
T = 3;               % Given T

%% Create fixed x_vals vector with 201 points from -1 to 1
x_vals = linspace(-1, 1, 201);

%% Initialize storage for unique rho sets per step
uniqueRho = cell(nSteps,1);

%% Define an inline function to check for duplicate rho values within tolerance.
% This function returns true if any set in cell array 'list' is equal (within tol) to new_rho.
isDuplicate = @(new_rho, list, tol) any(cellfun(@(x) all(abs(new_rho - x) < tol), list));


%% Loop over each step (jump)
for step = 1:nSteps
    fprintf('Processing %d step(s):\n', step);

    % Initialize unique rho set for this step if not already
    if isempty(uniqueRho{step})
        uniqueRho{step} = {};
    end
    
    % Combined loop for training, testing, and validation samples
    for m = 1:total_samples
        % Determine the data type and adjusted sample index based on m
        if m <= nTrain
            data_type = 'train';
            sample_index = m;
        elseif m <= nTrain + nTest
            data_type = 'test';
            sample_index = m - nTrain;
        else
            data_type = 'val';
            sample_index = m - nTrain - nTest;
        end
        
        success = false;
        while ~success
          
            % --- Generate a vector of "step" x thresholds ---
            % For each step, generate a random x value between -1 and 1 with 2-decimal precision.
            x_rand = zeros(1, step);
            for idx = 1:step
                x_temp = 2 * rand() - 1;
                x_temp = round(x_temp, 2);
                % Ensure x_temp is strictly between -0.95 and 0.95
                while x_temp <= -0.95 || x_temp >= 0.95
                    x_temp = 2 * rand() - 1;
                    x_temp = round(x_temp, 2);
                end
                x_rand(idx) = x_temp;
            end
            % Sort the x values in ascending order to define thresholds
            x_sorted = sort(x_rand);
            
            % --- Generate a vector of (step+1) rho values ---
            rho = zeros(1, step+1);
            % First rho value: uniformly chosen between 0.05 and 0.95
            rho(1) = 0.05 + 0.90 * rand;
            rho(1) = round(rho(1), 2);
            % Generate subsequent rho values based on the previous one
            for j = 2:(step+1)
                rho(j) = Inf;
                while (rho(j) < 0.05 || rho(j) > 0.95 || abs(rho(j)-rho(j-1)) <0.05)
                    diff_rho = normrnd(0, target_std);
                    rho(j) = rho(j-1) + diff_rho;
                    rho(j) = round(rho(j), 2);
                end
            end

            % --- Check for duplicate rho values ---
            % Use a tolerance of 1e-6 (since values are rounded to 2 decimals, they should match exactly)
            if ~isempty(uniqueRho{step}) && isDuplicate(rho, uniqueRho{step}, 1e-6)
                fprintf('Duplicate rho set detected for step %d, regenerating...\n', step);
                continue;  % Regenerate this sample
            end
            


            % --- Create the piecewise constant rho_vals vector ---
            % Initialize rho_vals with the same length as x_vals.
            rho_vals = zeros(size(x_vals));
            % Region 1: for all x_vals <= first threshold, assign rho(1)
            rho_vals(x_vals <= x_sorted(1)) = rho(1);
            % For intermediate regions (if more than one step)
            for i = 2:step
                rho_vals(x_vals > x_sorted(i-1) & x_vals <= x_sorted(i)) = rho(i);
            end
            % Region for x_vals greater than the last threshold: assign rho(step+1)
            rho_vals(x_vals > x_sorted(step)) = rho(step+1);
            
            % --- Construct the filename ---
            filename = sprintf('%s_%dstep_%d.mat', data_type, step, sample_index);
            
            % --- Create target folder if it does not exist ---
            if ~exist(data_type, 'dir')
                mkdir(data_type);
            end
            
            % Full file path inside the corresponding folder
            fullFilePath = fullfile(data_type, filename);
    
            % (Optional) Count the total number of files in the current folder
            files = dir;
            numFiles = sum(~[files.isdir]);
            fprintf('%s sample %d (%d step(s)), total files: %d\n', data_type, sample_index, step, numFiles);

            % --- Call the simulation function ---
            try
                % Here we pass:
                % x: the sorted thresholds (vector of length "step")
                % rho: the corresponding rho values (length "step+1")
                % x_vals: fixed 201-point vector from -1 to 1
                % rho_vals: piecewise constant vector based on x_sorted and rho
                [final_t, len_all_stored_data, all_stored_data, success] = ...
                    WFTcrowd_1(x_sorted, rho, x_vals', rho_vals', T, epsilon);
                if ~success
                    fprintf('Unsuccessful simulation for file %s, retrying...\n', filename);
                else
                    fprintf('File %s created successfully.\n', filename);
                    save(fullFilePath, 'all_stored_data');
                end
            catch ME
                if exist(fullFilePath, 'file')
                    delete(fullFilePath);
                end
                fprintf('Error occurred for file %s\nError: %s\n', filename, ME.message);
                success = false;
            end
        end % while ~success
    end % for m (combined samples)
end % for step (jumps)
