function data = generate_data(epsilon, grid_size, T, n_jumps_vec, n_train, n_test, n_val)
%GENERATE_DATA  Build a labelled dataset for the 1-D Hughes model.
%
%   data = generate_data(epsilon, grid_size, T, n_jumps_vec, n_train, n_test, n_val)
%   run in command window (example): data = generate_data(1/250, 200, 3, [1 2 3 4 5], 1000, 250, 250);
%   INPUTS
%     epsilon      mesh size, must be 1/N for integer N >= 50  (e.g. 1/250)
%     grid_size    square grid resolution for X and Y          (e.g. 200)
%     T            final simulation time                        (e.g. 3)
%     n_jumps_vec  vector of jump counts to include            (e.g. [1 2 3 4 5])
%     n_train      samples per jump count in training set
%     n_test       samples per jump count in test set
%     n_val        samples per jump count in validation set
%
%   OUTPUTS
%     data  struct with fields:
%       .train.X   grid_size x grid_size x (n_train*numel(n_jumps_vec))  single
%       .train.Y   same size as X
%       .test.X / .test.Y   similarly sized
%       .val.X  / .val.Y    similarly sized
%       .meta   parameters used to generate the dataset
%
%     X(:,:,i)  — grid_size x grid_size matrix:
%                 row 1 holds the initial-condition density profile,
%                 rows 2:end are zero.
%     Y(:,:,i)  — full grid_size x grid_size solution from the WFT solver.
%
%   Samples whose max cost-balance error exceeds 0.1 are rejected.
%   The dataset is also saved to a .mat file in the current directory.

n_per_jump = n_train + n_test + n_val;
n_types    = numel(n_jumps_vec);

total_train = n_train * n_types;
total_test  = n_test  * n_types;
total_val   = n_val   * n_types;

% Pre-allocate (single precision to keep file sizes manageable)
train_X = zeros(grid_size, grid_size, total_train, 'single');
train_Y = zeros(grid_size, grid_size, total_train, 'single');
test_X  = zeros(grid_size, grid_size, total_test,  'single');
test_Y  = zeros(grid_size, grid_size, total_test,  'single');
val_X   = zeros(grid_size, grid_size, total_val,   'single');
val_Y   = zeros(grid_size, grid_size, total_val,   'single');

% Per-sample metadata
train_jumps = zeros(total_train, 1, 'uint8');
test_jumps  = zeros(total_test,  1, 'uint8');
val_jumps   = zeros(total_val,   1, 'uint8');

% Per-sample grid accuracy errors (MAE vs 2x-finer reference)
train_errs = zeros(total_train, 1);
test_errs  = zeros(total_test,  1);
val_errs   = zeros(total_val,   1);

fprintf('\n=== generate_data: epsilon=1/%d  grid=%dx%d  T=%g ===\n', ...
        round(1/epsilon), grid_size, grid_size, T);

% =========================================================================
for jj = 1:n_types
    n_j = n_jumps_vec(jj);
    fprintf('\n[Jump count = %d]  target: %d train / %d test / %d val\n', ...
            n_j, n_train, n_test, n_val);

    buf_X   = zeros(grid_size, grid_size, n_per_jump, 'single');
    buf_Y   = zeros(grid_size, grid_size, n_per_jump, 'single');
    buf_err = zeros(n_per_jump, 1);

    collected = 0;
    attempts  = 0;
    rejected_err   = 0;
    rejected_crash = 0;

    while collected < n_per_jump
        attempts = attempts + 1;

        % ---- random initial condition ----------------------------------------
        % Jump locations: n_j points drawn uniformly in (-1, 1), then sorted.
        % Retry if any two are closer than 0.02 or within 0.02 of the boundary.
        x_ic = sort(-1 + 2 * rand(1, n_j));
        if any(x_ic <= -0.98) || any(x_ic >= 0.98)
            continue;
        end
        if n_j > 1 && any(diff(x_ic) < 0.02)
            continue;
        end

        % Densities: n_j+1 values drawn uniformly in (0.02, 0.95) to stay
        % away from the singularity at rho=1 and from rho=0.
        rho_ic = 0.02 + 0.93 * rand(1, n_j + 1);

        % ---- run WFT solver --------------------------------------------------
        try
            [rho_grid, E_max, grid_err] = solve_wft(x_ic, rho_ic, T, epsilon, grid_size);
        catch
            rejected_crash = rejected_crash + 1;
            continue;
        end

        % ---- quality filter --------------------------------------------------
        if isnan(E_max) || isinf(E_max) || E_max > 0.1
            rejected_err = rejected_err + 1;
            continue;
        end

        % ---- store -----------------------------------------------------------
        collected = collected + 1;

        X = zeros(grid_size, grid_size, 'single');
        X(1, :) = single(rho_grid(1, :));   % IC in first row, rest zero
        buf_X(:, :, collected) = X;
        buf_Y(:, :, collected) = single(rho_grid);
        buf_err(collected)     = grid_err;

        if mod(collected, max(1, floor(n_per_jump/10))) == 0 || collected == n_per_jump
            fprintf('  %3d / %d  (attempts %d | rejected: %d err, %d crash)\n', ...
                    collected, n_per_jump, attempts, rejected_err, rejected_crash);
        end
    end

    fprintf('  Done. Total attempts: %d  |  rejection rate: %.1f%%\n', ...
            attempts, 100 * (1 - n_per_jump / attempts));

    % ---- split into train / test / val --------------------------------------
    t_off = (jj-1) * n_train;
    e_off = (jj-1) * n_test;
    v_off = (jj-1) * n_val;

    i_tr  = 1            : n_train;
    i_te  = n_train+1    : n_train+n_test;
    i_va  = n_train+n_test+1 : n_per_jump;

    train_X(:, :, t_off+1 : t_off+n_train) = buf_X(:, :, i_tr);
    train_Y(:, :, t_off+1 : t_off+n_train) = buf_Y(:, :, i_tr);
    test_X(:,  :, e_off+1 : e_off+n_test)  = buf_X(:, :, i_te);
    test_Y(:,  :, e_off+1 : e_off+n_test)  = buf_Y(:, :, i_te);
    val_X(:,   :, v_off+1 : v_off+n_val)   = buf_X(:, :, i_va);
    val_Y(:,   :, v_off+1 : v_off+n_val)   = buf_Y(:, :, i_va);

    train_jumps(t_off+1 : t_off+n_train) = n_j;
    test_jumps( e_off+1 : e_off+n_test)  = n_j;
    val_jumps(  v_off+1 : v_off+n_val)   = n_j;

    train_errs(t_off+1 : t_off+n_train) = buf_err(i_tr);
    test_errs( e_off+1 : e_off+n_test)  = buf_err(i_te);
    val_errs(  v_off+1 : v_off+n_val)   = buf_err(i_va);
end

% =========================================================================
% Pack output struct
% =========================================================================
data.train.X      = train_X;
data.train.Y      = train_Y;
data.train.n_jumps = train_jumps;
data.test.X       = test_X;
data.test.Y       = test_Y;
data.test.n_jumps  = test_jumps;
data.val.X        = val_X;
data.val.Y        = val_Y;
data.val.n_jumps   = val_jumps;

data.meta.epsilon     = epsilon;
data.meta.grid_size   = grid_size;
data.meta.T           = T;
data.meta.n_jumps_vec = n_jumps_vec;
data.meta.n_train     = n_train;
data.meta.n_test      = n_test;
data.meta.n_val       = n_val;
data.meta.grid_mae    = struct('train', train_errs, 'test', test_errs, 'val', val_errs);

% =========================================================================
% Grid accuracy report
% =========================================================================
all_errs = [train_errs; test_errs; val_errs];
fprintf('\n=== Grid accuracy: MAE(grid_size vs 2x-finer reference) ===\n');
fprintf('  %-10s  mean = %.4e   max = %.4e   min = %.4e\n', ...
        'Overall', mean(all_errs),   max(all_errs),   min(all_errs));
fprintf('  %-10s  mean = %.4e   max = %.4e\n', 'Train',  mean(train_errs), max(train_errs));
fprintf('  %-10s  mean = %.4e   max = %.4e\n', 'Test',   mean(test_errs),  max(test_errs));
fprintf('  %-10s  mean = %.4e   max = %.4e\n', 'Val',    mean(val_errs),   max(val_errs));
fprintf('\n  Per jump count (train):\n');
for jj = 1:n_types
    n_j   = n_jumps_vec(jj);
    emask = train_errs(train_jumps == n_j);
    fprintf('    %d jump(s):  mean = %.4e   max = %.4e\n', n_j, mean(emask), max(emask));
end

% =========================================================================
% Save to file
% =========================================================================
fname = sprintf('hughes_data_eps%d_grid%d_T%g_%djumptypes.mat', ...
                round(1/epsilon), grid_size, T, n_types);
fprintf('\nSaving dataset to  %s  ...  ', fname);
save(fname, 'data', '-v7.3');
fprintf('done.\n');

end


% =========================================================================
%  LOCAL FUNCTION: headless WFT solver (no plotting)
%  Returns the reconstructed grid and the max cost-balance error.
% =========================================================================
function [rho_grid, E_max, grid_err] = solve_wft(x, rho, T, epsilon, grid_size)

c = @(u) 1./(1-u);

rho  = [0, id(rho, epsilon), 0];
xi   = turningpoint([-1, x, 1], rho);
xl   = [-1, x(x < xi)];
xr   = [x(x > xi), 1];
rhol = rho(1 : length(xl)+1);
rhor = rho(end - length(xr) : end);

t  = 0;
Tc = T;
E  = 0;

[xl, xi, xr, rhol, rhor, sl, si, sr] = PdR(xl, xi, xr, rhol, rhor, epsilon);

tw   = 0;
xw   = {[xl, xi, xr]};
sw   = {[sl, si, sr]};
rhow = {[rhol, rhor]};

while t <= T
    x_all = [xl, xi, xr];
    s_all = [sl, si, sr];

    denom    = s_all(1:end-1) - s_all(2:end);
    gap      = x_all(2:end)   - x_all(1:end-1);
    collider = gap ./ denom;
    tc       = min(collider(collider > 0));
    if isempty(tc), tc = Inf; end

    E(end+1) = sum(c(rhol(2:end)) .* ([xl(2:end), xi] - xl)) ...   %#ok<AGROW>
             - sum(c(rhor(1:end-1)) .* (xr - [xi, xr(1:end-1)]));

    if tc == Inf || tc > Tc
        break;
    end

    t  = t  + tc;
    Tc = Tc - tc;
    xl = xl + tc * sl;
    xi = xi + tc * si;
    xr = xr + tc * sr;

    % Suppress coincident fronts (left of xi)
    xf = [xl, xi];  rhof = [rhol, rhor(1)];
    idx = find(abs(xf(1:end-1) - xf(2:end)) < 1e-14) + 1;
    xf(idx) = [];  rhof(idx) = [];
    xl = xf(1:end-1);  xi = xf(end);  rhol = rhof(1:end-1);

    % Suppress coincident fronts (right of xi)
    xf = [xi, xr];  rhof = [rhol(end), rhor];
    idx = find(abs(xf(1:end-1) - xf(2:end)) < 1e-14) + 1;
    xf(idx) = [];  rhof(idx) = [];
    xr = xf(2:end);  xi = xf(1);  rhor = rhof(2:end);

    [xl, xi, xr, rhol, rhor, sl, si, sr] = PdR(xl, xi, xr, rhol, rhor, epsilon);

    tw(end+1)   = t;               %#ok<AGROW>
    xw{end+1}   = [xl, xi, xr];   %#ok<AGROW>
    sw{end+1}   = [sl, si, sr];    %#ok<AGROW>
    rhow{end+1} = [rhol, rhor];    %#ok<AGROW>
end

tw(end+1) = T;
E_max = max(abs(E));

[rho_grid, ~, ~] = reconstruct(rhow, xw, sw, tw, T, grid_size, grid_size);

% ---- grid accuracy: compare to 2x-finer reconstruction -------------------
N_ref = 2 * grid_size;
[rho_ref, ~, ~] = reconstruct(rhow, xw, sw, tw, T, N_ref, N_ref);

% Block-average the 2x grid down to grid_size x grid_size.
% Time direction: average pairs of rows.
tmp = reshape(rho_ref, 2, grid_size, N_ref);   % (2, gs, 2*gs)
tmp = squeeze(mean(tmp, 1));                    % (gs, 2*gs)
% Space direction: average pairs of columns.
tmp = reshape(tmp, grid_size, 2, grid_size);    % (gs, 2, gs)
rho_ref_ds = squeeze(mean(tmp, 2));             % (gs, gs)

grid_err = mean(abs(rho_grid(:) - rho_ref_ds(:)));

end
