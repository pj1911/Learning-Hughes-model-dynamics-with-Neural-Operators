%
% WFTCROWD v1.1
% Initial Sofware by GOATIN P. and MIMAULT M., Copyright (C) Inria (Opale team), 
% All Rights Reserved, 2012
%
function [] = profil(x,u,s,tc)

%Profil plotting between each interaction

dt = 0.01;
vt = 0:dt:tc;

if vt(end)~=tc
    vt = [vt tc];
end

if dt>tc
    dt = tc;
end

for n = 2:length(vt)
    stairs(x+s*vt(n),u)
    axis([-1 1 0 1])
    if dt>=10^-2
        pause(0.01)
    elseif dt<10^-2 && dt>10^-3
        pause(0.001)
    end
end