function run_a_sample(x, rho, T, epsilon)
%RUN_A_SAMPLE  Run the 1-D Hughes model WFT solver and compare to grid.
%
%   run_a_sample()
%   run_a_sample(x, rho, T, epsilon)
%
%   INPUTS (all optional, defaults match WFTcrowd hardcoded values)
%     x        row vector of interface positions in (-1,1), e.g. [-0.543 -0.387]
%     rho      density in each region, length(x)+1 values, e.g. [0.867 0.839 0.456]
%     T        final time, e.g. 3
%     epsilon  mesh size 1/N for integer N, e.g. 1/250
%
%   The function:
%     1. Runs the wavefront-tracking (WFT) solver using PdR / turningpoint / id.
%     2. Calls reconstruct() to build a uniform (x,t) grid matrix.
%     3. Plots three panels side by side:
%          Panel 1  WFT space-time diagram   (exact patch representation)
%          Panel 2  Grid reconstruction      (imagesc of the Nt x Nx matrix)
%          Panel 3  Absolute error           (fine grid vs coarse grid,
%                                             illustrating resolution effects)

% ---- defaults ---------------------------------------------------------------
if nargin < 1 || isempty(x),       x       = [-0.743, -0.387, 0, 0.8]; end
if nargin < 2 || isempty(rho),     rho     = [0.967, 0.839, 0.456, 0.95, 0.7]; end
if nargin < 3 || isempty(T),       T       = 3; end
if nargin < 4 || isempty(epsilon), epsilon = 1/350; end

c = @(u) 1./(1-u);

% =============================================================================
% 1.  INITIALISE
% =============================================================================
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

% Storage arrays — consistent indexing:
%   interval k spans [tw(k), tw(k+1)]
%   xw{k}, sw{k}, rhow{k} all describe the state at the START of interval k
tw   = 0;
xw   = {[xl, xi, xr]};
sw   = {[sl, si, sr]};
rhow = {[rhol, rhor]};

% Patch data for plotting (accumulated during the loop)
patch_xv = {};   % 4 x (N-1) x-corners for each strip
patch_tv = {};   % 4 x (N-1) t-corners for each strip
patch_cv = {};   % 1 x (N-1) colour (density/epsilon) for each strip

% =============================================================================
% 2.  WFT MAIN LOOP
% =============================================================================
while t <= T
    x_all   = [xl, xi, xr];
    s_all   = [sl, si, sr];
    rho_all = [rhol, rhor];

    % Next wave-collision time
    denom    = s_all(1:end-1) - s_all(2:end);
    gap      = x_all(2:end)   - x_all(1:end-1);
    collider = gap ./ denom;
    tc       = min(collider(collider > 0));
    if isempty(tc), tc = Inf; end

    % Cost-integral balance error (diagnostic)
    E(end+1) = sum(c(rhol(2:end)) .* ([xl(2:end), xi] - xl)) ...
             - sum(c(rhor(1:end-1)) .* (xr - [xi, xr(1:end-1)])); %#ok<AGROW>

    % Clamp end time to T
    if tc == Inf || tc > Tc
        t_end = T;
        dt_p  = Tc;
    else
        t_end = t + tc;
        dt_p  = tc;
    end

    % Accumulate patch data for this strip
    x2 = x_all + s_all * dt_p;
    tp = [t, t_end, t_end, t]' * ones(1, length(x_all)-1);
    patch_xv{end+1} = [x_all(1:end-1); x2(1:end-1); x2(2:end); x_all(2:end)]; %#ok<AGROW>
    patch_tv{end+1} = tp;                                                        %#ok<AGROW>
    patch_cv{end+1} = rho_all(2:end-1);                                         %#ok<AGROW>

    % Exit if no further collisions within [t, T]
    if tc == Inf || tc > Tc
        break;
    end

    % ---- advance state to next collision ----
    t  = t  + tc;
    Tc = Tc - tc;

    xl = xl + tc * sl;
    xi = xi + tc * si;
    xr = xr + tc * sr;

    % Suppress coincident fronts on the left of xi
    xf   = [xl, xi];
    rhof = [rhol, rhor(1)];
    idx  = find(abs(xf(1:end-1) - xf(2:end)) < 1e-14) + 1;
    xf(idx) = []; rhof(idx) = [];
    xl = xf(1:end-1); xi = xf(end); rhol = rhof(1:end-1);

    % Suppress coincident fronts on the right of xi
    xf   = [xi, xr];
    rhof = [rhol(end), rhor];
    idx  = find(abs(xf(1:end-1) - xf(2:end)) < 1e-14) + 1;
    xf(idx) = []; rhof(idx) = [];
    xr = xf(2:end); xi = xf(1); rhor = rhof(2:end);

    % New Riemann solve at each wavefront
    [xl, xi, xr, rhol, rhor, sl, si, sr] = PdR(xl, xi, xr, rhol, rhor, epsilon);

    % Store state for this new interval
    tw(end+1)   = t;          %#ok<AGROW>
    xw{end+1}   = [xl, xi, xr]; %#ok<AGROW>
    sw{end+1}   = [sl, si, sr];  %#ok<AGROW>
    rhow{end+1} = [rhol, rhor];  %#ok<AGROW>
end

tw(end+1) = T;   % sentinel — marks the end of the last interval

fprintf('Simulation done.  Intervals: %d   Max cost-balance error: %.3e\n', ...
        length(xw), max(abs(E)));

% =============================================================================
% 3.  GRID RECONSTRUCTION (fine and coarse for error panel)
% =============================================================================
Nx_fine   = 200;
Nt_fine   = 200;
Nx_coarse = 100;
Nt_coarse = 100;

[rho_fine,   x_fine,   t_fine]   = reconstruct(rhow, xw, sw, tw, T, Nx_fine,   Nt_fine);
[rho_coarse, x_coarse, t_coarse] = reconstruct(rhow, xw, sw, tw, T, Nx_coarse, Nt_coarse);

% Upsample coarse grid to fine resolution (nearest-neighbour) for error map
rho_up = interp2(x_coarse, t_coarse, rho_coarse, x_fine, t_fine, 'nearest', 0);
abs_err = abs(rho_fine - rho_up);

% =============================================================================
% 4.  PLOT
% =============================================================================
figure('Units','pixels', 'Position',[50 50 1350 460]);

% ---- Panel 1: exact WFT space-time patch diagram ----------------------------
ax1 = subplot(1, 3, 1);
hold(ax1, 'on');
for k = 1:length(patch_xv)
    p = patch(patch_xv{k}, patch_tv{k}, 'r', 'Parent', ax1);
    set(p, 'FaceColor','flat', ...
           'CData',        patch_cv{k}, ...
           'CDataMapping','scaled', ...
           'EdgeColor',   'none');
end
set(ax1, 'CLim', [0, 1]);
colorbar(ax1);
xlabel(ax1, 'x');  ylabel(ax1, 't');
title(ax1, 'WFT solver — exact space-time diagram');
axis(ax1, [-1, 1, 0, T]);

% ---- Panel 2: uniform grid reconstruction -----------------------------------
ax2 = subplot(1, 3, 2);
imagesc(ax2, x_fine, t_fine, rho_fine);
set(ax2, 'YDir','normal', 'CLim', [0, 1]);
colorbar(ax2);
xlabel(ax2, 'x');  ylabel(ax2, 't');
title(ax2, sprintf('Grid reconstruction  (%d\\times%d)', Nx_fine, Nt_fine));
axis(ax2, [-1, 1, 0, T]);

% ---- Panel 3: absolute error (fine vs coarse) --------------------------------
ax3 = subplot(1, 3, 3);
imagesc(ax3, x_fine, t_fine, abs_err);
set(ax3, 'YDir','normal');
colorbar(ax3);
xlabel(ax3, 'x');  ylabel(ax3, 't');
title(ax3, sprintf('|fine (%d\\times%d) - coarse (%d\\times%d) upsampled|', ...
      Nx_fine, Nt_fine, Nx_coarse, Nt_coarse));
axis(ax3, [-1, 1, 0, T]);

colormap(jet);
sgtitle(sprintf('Hughes model 1-D  —  WFT vs grid reconstruction  (\\epsilon = 1/%d)', ...
        round(1/epsilon)));

end
