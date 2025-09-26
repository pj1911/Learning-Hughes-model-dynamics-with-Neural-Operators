%
% WFTCROWD v1.1
% Initial Sofware by GOATIN P. and MIMAULT M., Copyright (C) Inria (Opale team), 
% All Rights Reserved, 2012
%
function [stored_values, last_saved_time] = profil(x,u,s,tc, tp, tp_target, ttn, last_time)

%Profil plotting between each interaction

dt = 0.0005;
vt = 0:dt:tc;

if vt(end)~=tc
    vt = [vt tc];
end

if dt>tc
    dt = tc;
end

% Initialize a cell array to store k and u vectors
stored_values = {};
eps = 0.000001;
last_saved_time = last_time;
row_index = 1;
delta_t = (tp_target-tp ) / (length(vt)-1);
for n = 2:length(vt)
    k = x+s*vt(n);
    
    time_till_now = ttn +  vt(n);

    k = round(k,2);
    u = round(u,2);
    

    [u_unique, k_unique] = unique_xy_coords(u,k);

    

    if (time_till_now - last_saved_time) > 0.005-eps
        % Store k, u, and tp only if the time increment is >= 0.01
        stored_values{row_index, 1} = k_unique;  % store k in the first column
        stored_values{row_index, 2} = u_unique;  % store u in the second column
        stored_values{row_index, 3} = round(tp + ((tp_target-tp ) / (length(vt)-1)), 2);  % store tp rounded to 2 decimals
        % stored_values{row_index, 4} = round ( findTurningPoint(u_unique ), 2);
        stored_values{row_index, 4} = round ( compute_psi_single(u_unique ), 2);

        last_saved_time = time_till_now;
        row_index = row_index+1 ;
    end
    tp = tp + delta_t;

end
return
end
