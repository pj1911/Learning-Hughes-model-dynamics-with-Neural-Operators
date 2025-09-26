%
% WFTCROWD v1.1
% Initial Sofware by GOATIN P. and MIMAULT M., Copyright (C) Inria (Opale team), 
% All Rights Reserved, 2012
%
function uf = id(u,epsilon)

%Procedure used to round up each value of u according to the epsilon step

ulin = 0:epsilon:1;

if ulin(end)~=1
    epsilon = 1/length(ulin);
    ulin = 0:epsilon:1;
end

ulin(end) = ulin(end-1);

for j = 1:length(u)
    for i = 1:length(ulin)
        if ulin(i)-epsilon/2<=u(j) && u(j)<ulin(i)+epsilon/2
            uf(j) = ulin(i);
        elseif 1-epsilon/2<=u(j) && u(j)<1+epsilon/2
            uf(j) = ulin(i);
        end
    end
end