function [u_unique,k_unique] = unique_xy_coords(rho,x_vals)

    % Step 3: Process each unique value of k
    x_range = round(-1:0.01:1, 2)'; % Predefined range
    rho_new = nan(size(x_range));     % Initialize y_new with NaN
    threshold  = 0.001;

    current_value = NaN;

    for i = 1:length(x_range)

        if ~isnan(rho_new(i)) && all(abs(x_vals - x_range(i)) > threshold)
            continue;
        end

        if sum(abs(x_vals - x_range(i)) <= threshold) > 1
            rho_values = rho(abs(x_vals - x_range(i)) <= threshold);

            min_rho  = min(rho_values);
            last_rho = rho_values(end);

            % Check if min_rho and last_rho are the same
            if abs(min_rho - last_rho) <= threshold
                current_value = min_rho; % Keep only one as the current value
                next_value = NaN;       % No next value needed
            else
                current_value = min_rho; % Keep min as current value
                next_value = last_rho;  % Keep last as next value
            end
        else
            if any(abs(x_vals - x_range(i)) <= threshold)
                index = find(abs(x_vals - x_range(i)) <= threshold, 1);
                current_value = rho(index);
                next_value = NaN; % No next value needed
            end
        end
        rho_new(i) = current_value;
        if ~isnan(next_value) && i + 1 < length(rho_new)  %&& all(abs(x_vals - x_range(i + 1)) > threshold)
            rho_new(i + 1) = next_value;
            current_value = next_value;
            next_value = NaN;
        end


    end

    u_unique = rho_new;
    k_unique = x_range;
end
