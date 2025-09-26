%
% WFTCROWD v1.1
% Initial Sofware by GOATIN P. and MIMAULT M., Copyright (C) Inria (Opale team), 
% All Rights Reserved, 2012
%
function xm = turningpoint(x,u)

c = @(u) 1./(1-u);

A =(x(2:end)-x(1:end-1)).*c(u(2:end-1));
At = sum(A);
Al = 0;
Ar = 0;
i = 1;
xl = x(1);
xr = x(end);

while Al+(x(i+1)-x(i))*c(u(i+1))<At/2
    Al = Al+(x(i+1)-x(i))*c(u(i+1));
    i = i+1;
    xl = x(i);
end


i = length(x)-1;

while Ar+(x(i+1)-x(i))*c(u(i+1))<At/2
    Ar = Ar+(x(i+1)-x(i))*c(u(i+1));
    i = i-1;
    xr = x(i+1);
end

xm = ((xr-xl)*(Ar-Al)/(At-Ar-Al)+xl+xr)/2;
