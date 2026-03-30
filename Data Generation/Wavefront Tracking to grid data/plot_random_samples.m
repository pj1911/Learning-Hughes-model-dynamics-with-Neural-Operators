function plot_random_samples(source)
%PLOT_RANDOM_SAMPLES  Plot 2 random training samples for each jump count.
%
%   plot_random_samples(data)      pass the struct returned by generate_data
%   plot_random_samples(filename)  pass the .mat filename as a string
%   run in command window (exampe): plot_random_samples('hughes_data_eps250_grid200_T3_5jumptypes.mat');
%   Displays a grid of subplots:  rows = jump counts,  cols = 2 samples.
%   Only training data (Y, the full solution) is shown.

% ---- load data --------------------------------------------------------------
if ischar(source) || isstring(source)
    tmp  = load(source, 'data');
    data = tmp.data;
else
    data = source;
end

n_jumps_vec = data.meta.n_jumps_vec;
n_types     = numel(n_jumps_vec);
grid_size   = data.meta.grid_size;
T           = data.meta.T;

n_cols = 2;
n_rows = n_types;

figure('Units','pixels', 'Position', [50 50 n_cols*340 n_rows*280]);

for jj = 1:n_types
    n_j = n_jumps_vec(jj);

    % Indices in the training set that belong to this jump count
    idx_pool = find(data.train.n_jumps == n_j);

    if numel(idx_pool) < 2
        warning('Fewer than 2 training samples for jump count %d — skipping.', n_j);
        continue;
    end

    % Pick 2 without replacement
    chosen = idx_pool(randperm(numel(idx_pool), 2));

    for col = 1:2
        sp = subplot(n_rows, n_cols, (jj-1)*n_cols + col);

        Y = double(data.train.Y(:, :, chosen(col)));

        imagesc(sp, linspace(-1,1,grid_size), linspace(0,T,grid_size), Y);
        set(sp, 'YDir', 'normal', 'CLim', [0 1]);
        colormap(sp, jet);
        colorbar(sp);

        xlabel(sp, 'x');
        ylabel(sp, 't');
        title(sp, sprintf('%d jump(s) — sample %d', n_j, chosen(col)));
    end
end

sgtitle(sprintf('Training set — 2 random samples per jump count  (\\epsilon = 1/%d,  grid %d\\times%d)', ...
        round(1/data.meta.epsilon), grid_size, grid_size));

end
