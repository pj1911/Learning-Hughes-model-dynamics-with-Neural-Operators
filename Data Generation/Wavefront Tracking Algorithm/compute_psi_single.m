function turning_point_x = compute_psi_single(density)


    % Define the cost function
    cost_function = @(rho) 1 ./ (1 - rho);

    % Define dx and corresponding x values
 
    n = 201 ;
    x_values = linspace(-1, 1, n)'; % Column vector of x-values

    % Compute the cost for the given density
    cost = cost_function(density); % (n x 1)

    % Compute cumulative cost to the left and right of each space point
    left_cost_cumsum = cumsum(cost); % Cumulative sum from left
    right_cost_cumsum = flip(cumsum(flip(cost))); % Cumulative sum from right

    % Compute the absolute difference between left and right cumulative costs
    cost_diff = abs(left_cost_cumsum - right_cost_cumsum);

    % Find the index of the minimum cost difference
    [~, turning_point_idx] = min(cost_diff);

    % Return the x-value corresponding to the turning point
    turning_point_x = x_values(turning_point_idx);
end