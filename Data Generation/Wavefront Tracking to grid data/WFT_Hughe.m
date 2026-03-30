% This part is outside the function and calls WFT_Hughes
x = [0];  % Discontinuity point at x = 0
rho = [0, 0.9];  % Initial density: 0 to the left, 0.9 to the right
T = 3;  % Total simulation time
epsilon = 1/250;  % Mesh size

% Call the WFT_Hughes function
WFT_Hughes(x, rho, T, epsilon)

% The WFT_Hughes function
function [] = WFT_Hughes(x, rho, T, epsilon)
    close all;
    tic;

    % Define cost function and velocity
    c = @(u) 1./(1-u);    % Cost function
    v = @(u) (1-u);       % Velocity function
    f = @(u) u .* v(u);   % Flux function

    % Initialize density with boundary conditions
    rho = [0 rho 0];  % Add boundary conditions for rho
    
    % Determine initial turning point xi
    xi = turningpoint(x, rho, c);  % Pass x, rho, and c to turningpoint
    
    xl = [-1 x(x<xi)];   % Splitting interfaces to the left
    xr = [x(x>xi) 1];    % Splitting interfaces to the right
    rhol = rho(1:length(xl)+1);      % Left density
    rhor = rho(end-length(xr):end);  % Right density
    
    t = 0;
    Tc = T;
    E = 0;  % Cost integral balance error

    % Initialize plot window
    figure('Units', 'pixels', 'Position', [75 75 850 375]);

    % Main simulation loop
    while t <= T
        % Update position and solver state using wavefront tracking
        [xl, xi, xr, rhol, rhor, sl, si, sr] = PdR(xl, xi, xr, rhol, rhor, epsilon);
        
        % Cost balance error
        E = [E compute_cost_error(rhol, xl, rhor, xr, xi, c)];
        
        % Plot current solution
        subplot(1, 2, 1);
        hold on;
        % Plot the densities as patch regions
        plot_density(t, xl, xi, xr, rhol, rhor, sl, si, sr, epsilon);
        
        % Update the time step based on wave collision times
        collider = compute_collider(xl, xi, xr, sl, si, sr);
        tc = min(collider(collider > 0));  % Find the closest collision time

        % Handle if no further collisions happen
        if tc == Inf
            break;
        end
        
        % Plot the updated positions and profiles after time step tc
        subplot(1, 2, 1); % For visualizing the density profile
        plot_density(t, xl, xi, xr, rhol, rhor, sl, si, sr, epsilon);

        % Update position and density for the next iteration
        [xl, xi, xr, rhol, rhor] = update_position_density(tc, xl, xi, xr, rhol, rhor, sl, si, sr);
        
        % Update time
        t = t + tc;
        Tc = Tc - tc;
        pause(0.1);  % Add a pause to see the plot updates
    end

    % Final plot and save results
    disp('Cost integral balance error');
    max(abs(E));
    save_results(xl, xi, xr, rhol, rhor, t, epsilon);
end

% --- Helper Functions (These are necessary for the simulation) ---

% Plotting the density profile at each time step
function plot_density(t, xl, xi, xr, rhol, rhor, sl, si, sr, epsilon)
    % Plot the density profile as patch regions
    x = [xl xi xr];
    rho = [rhol rhor];
    for i = 1:length(x)-1
        patch([x(i) x(i+1) x(i+1) x(i)], [t t t+1 t+1], rho(i)*[1 1 1 1], 'EdgeColor', 'none');
    end
    plot([xi; xi+si*t],[t; t+1],'w','LineWidth',2);  % Show turning point
    caxis([0 1]);
    colorbar;
    title(['Density Profile at Time t = ', num2str(t)]);
end

% Function to compute cost integral error
function error = compute_cost_error(rhol, xl, rhor, xr, xi, c)
    % Compute the cost integral balance error
    left_cost = sum(c(rhol(2:end)) .* (xl(2:end) - xl(1:end-1)));
    right_cost = sum(c(rhor(1:end-1)) .* (xr(2:end) - xr(1:end-1)));
    error = left_cost - right_cost;
end

% Function to update position and density after each step
function [xl, xi, xr, rhol, rhor] = update_position_density(tc, xl, xi, xr, rhol, rhor, sl, si, sr)
    xl = xl + tc * sl;  % Update left position
    xi = xi + tc * si;  % Update turning point
    xr = xr + tc * sr;  % Update right position
    % Update densities as required (simplified for now)
end

% Function to compute the next wave collision time
function collider = compute_collider(xl, xi, xr, sl, si, sr)
    % Compute possible collision times between waves
    x = [xl xi xr];
    s = [sl si sr];
    collider = (x(2:end) - x(1:end-1)) ./ (s(1:end-1) - s(2:end));  % Find collider times
end

% Function to save the simulation results
function save_results(xl, xi, xr, rhol, rhor, t, epsilon)
    epsilonstr = int2str(1/epsilon);
    name = strcat('Hughes_Simulation_', epsilonstr);
    save(name, 'xl', 'xi', 'xr', 'rhol', 'rhor', 't');
end