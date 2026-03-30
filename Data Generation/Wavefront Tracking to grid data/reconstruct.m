function [rho_grid, x_grid, t_grid] = reconstruct(rhow, xw, sw, tw, T, Nx, Nt)
%RECONSTRUCT  Convert WFT solver output into a uniform (x,t) grid matrix.
%
%   [rho_grid, x_grid, t_grid] = reconstruct(rhow, xw, sw, tw, T)
%   [rho_grid, x_grid, t_grid] = reconstruct(rhow, xw, sw, tw, T, Nx, Nt)
%
%   INPUTS
%     rhow   cell array  density vectors, one per time interval.
%                        rhow{k} has length length(xw{k})+1.
%                        The density in region [xw{k}(j), xw{k}(j+1)] is
%                        rhow{k}(j+1)  (rhow{k}(1) and rhow{k}(end) are the
%                        left/right boundary values, typically 0).
%     xw     cell array  wavefront position vectors at the START of each
%                        interval. xw{k}(1) = -1, xw{k}(end) = 1.
%     sw     cell array  wavefront speed vectors, same size as xw.
%     tw     vector      collision/interaction times including t=0 at tw(1)
%                        and t=T at tw(end). length(tw) = length(xw)+1.
%     T      scalar      final simulation time
%     Nx     integer     number of spatial grid points  (default 200)
%     Nt     integer     number of temporal grid points (default 200)
%
%   OUTPUTS
%     rho_grid   Nt x Nx  density matrix on the uniform grid
%     x_grid     1  x Nx  spatial grid on [-1, 1]
%     t_grid     Nt x 1   temporal grid on [0, T]
%
%   CONVENTION (matches run_a_sample output)
%     Interval k spans [tw(k), tw(k+1)].
%     At time t in that interval the j-th wavefront is at
%       xw{k}(j) + sw{k}(j) * (t - tw(k)).
%     The density between consecutive wavefronts is piecewise constant and
%     is evaluated exactly using step-function (previous-neighbour)
%     interpolation.

if nargin < 6 || isempty(Nx), Nx = 200; end
if nargin < 7 || isempty(Nt), Nt = 200; end

x_grid   = linspace(-1, 1, Nx);
t_grid   = linspace(0, T, Nt)';
rho_grid = zeros(Nt, Nx);

n_intervals = length(xw);   % number of intervals with data

for it = 1:Nt
    t = t_grid(it);

    % Find interval index k : tw(k) <= t < tw(k+1)
    k = find(tw(1:n_intervals) <= t + 1e-12, 1, 'last');
    if isempty(k), k = 1; end
    k = min(k, n_intervals);

    % Advance wavefront positions from tw(k) to t
    dt    = max(0, t - tw(k));
    x_now = xw{k} + sw{k} * dt;   % 1 x N_fronts, should span [-1,1]

    % Evaluate piecewise-constant density on x_grid.
    % rhow{k}(j+1) is the density in [x_now(j), x_now(j+1)].
    % interp1 with 'previous' returns the value at the largest knot <= query,
    % so mapping knots x_now -> rhow{k}(2:end) delivers rhow{k}(j+1) for any
    % query in [x_now(j), x_now(j+1)).
    rv = rhow{k};

    % Duplicate wavefront positions arise when two fronts just collided.
    % unique(...,'last') keeps the last occurrence of each duplicate x value,
    % which corresponds to the density on the outgoing side of the collision.
    rv_region = rv(2:end);          % length == length(x_now)
    [x_now, ia] = unique(x_now, 'last');
    rv_region   = rv_region(ia);

    rho_grid(it, :) = interp1(x_now, rv_region, x_grid, 'previous', 0);
end
end
