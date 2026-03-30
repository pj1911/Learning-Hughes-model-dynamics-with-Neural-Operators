%
% WFTCROWD v1.1
% Initial Sofware by GOATIN P. and MIMAULT M., Copyright (C) Inria (Opale team), 
% All Rights Reserved, 2012
%
function uf = id(u,epsilon) % rho, epsilon

%Procedure used to round up each value of u according to the epsilon step

ulin = 0:epsilon:1;

if ulin(end)~=1  % Ensures that ulin reaches exactly 1 by slightly adjusting epsilon if necessary.
    epsilon = 1/length(ulin);
    ulin = 0:epsilon:1;
end

ulin(end) = ulin(end-1);

for j = 1:length(u)                                        % loop over all inital density values
    for i = 1:length(ulin)                                      % This depends on epsilon
        if ulin(i)-epsilon/2<=u(j) && u(j)<ulin(i)+epsilon/2
            uf(j) = ulin(i);
        elseif 1-epsilon/2<=u(j) && u(j)<1+epsilon/2
            uf(j) = ulin(i);
        end
    end
end