%
% WFTCROWD v1.1
% Initial Sofware by GOATIN P. and MIMAULT M., Copyright (C) Inria (Opale team), 
% All Rights Reserved, 2012
%
function [xl, xm, xr, rhol, rhor, sl, sm, sr] = PdR(xl,xm,xr,rhol,rhor,epsilon)
%Riemann Solveur
v = @(u) 1-u;
f = @(u) u.*v(u);
c = @(u) 1./(1-u);
q = @(u) -1./(1-u)-2.*log(1-u);

xf = [];
rhof = rhol(1);
opts = optimset('Display', 'off');

for i = 1:length(xl)
    if rhol(i)>rhol(i+1) %Shock
        w = rhol(i+1);
    else
        w = rhol(i):epsilon:rhol(i+1); %RarefactionWave
        if w(end)~=rhol(i+1)
            w = [w rhol(i+1)];
        end
        w(1) = [];
    end
    xf = [xf xl(i)*ones(1,length(w))];
    rhof = [rhof w];
end
xl = xf;
rhol = rhof;
sl = (f(rhol(1:end-1))-f(rhol(2:end)))./(rhol(2:end)-rhol(1:end-1));
while ~isempty(sl) && sl(1)<0 && xl(1)==-1;
    sl(1) = [];
    rhol(1) = [];
    xl(1) = [];
end
sl = [0 sl];
rhol = [rhol(1) rhol];
xl = [-1 xl];


xf = [];
rhof = rhor(1);
for i = 1:length(xr) %Shock
    if rhor(i)<rhor(i+1)
        w = rhor(i+1);
    else
        w = rhor(i):-epsilon:rhor(i+1); %RarefactionWave
        if w(end)~=rhor(i+1)
            w = [w rhor(i+1)];
        end
        w(1) = [];
    end
    xf = [xf xr(i)*ones(1,length(w))];
    rhof = [rhof w];

end
xr = xf;
rhor = rhof;
sr = (f(rhor(2:end))-f(rhor(1:end-1)))./(rhor(2:end)-rhor(1:end-1));
while ~isempty(sr) && sr(end)>0 && xr(end)==1
    sr(end) = [];
    rhor(end) = [];
    xr(end) = [];
end
sr = [sr 0];
rhor = [rhor rhor(end)];
xr = [xr 1];

psistar = sum(sr.*(c(rhor(1:end-1))-c(rhor(2:end))))-sum(sl.*(c(...
    rhol(1:end-1))-c(rhol(2:end))));


%Solution around xi
xi = @(u) (f(u)+f(rhol(end)))/(u-rhol(end))*(c(u)+c(rhol(end)))+q(rhor(1))-q(u);
xi2 = @(u) (f(rhor(1))+f(u))/(rhor(1)-u)*(c(rhor(1))+c(u))+q(u)-q(rhol(end));
theta = @(u) (f(u)+f(rhol(end)))/(u-rhol(end))*(c(u)+c(rhol(end)))...
    -(f(u)-f(rhor(1)))/(u-rhor(1))*(c(u)-c(rhor(1)));
lambda = @(u) (f(u)+f(rhor(1)))/(rhor(1)-u)*(c(u)+c(rhor(1)))...
    -(f(rhol(end))-f(u))/(rhol(end)-u)*(c(rhol(end))-c(u));



if rhol(end)==rhor(1)
    if psistar<theta(0)
        %Rarefaction between ul and ur
        rhom = fsolve(@(u) theta(u)-psistar, 0,opts);
        rhom = id(rhom,epsilon);
        sm = (f(rhol(end))+f(rhom))/(rhom-rhol(end));
        if rhom~=rhor(1)
            sr = [(f(rhom)-f(rhor(1)))/(rhom-rhor(1)) sr];
            rhor = [rhom rhor];
            xr = [xm xr];
        end
        if  rhom==rhol(end)
            sm = (psistar+(v(rhol(end)))*(c(rhol(end))-1)...
                +(v(rhor(1)))*(1-c(rhor(1))))/2;
        end
    elseif psistar>lambda(0)
        %Shock x_l and x_m
        rhom = fsolve(@(u) lambda(u)-psistar, 0,opts);
        rhom = id(rhom,epsilon);
        sm = (f(rhor(1))+f(rhom))/(rhor(1)-rhom);
        if rhom~=rhol(end)
            sl = [sl -(f(rhol(end))-f(rhom))/(rhol(end)-rhom)];
            rhol = [rhol rhom];
            xl = [xl xm];
        end
        if  rhom==rhor(1)
            sm = (psistar+(v(rhol(end)))*(c(rhol(end))-1)...
                +(v(rhor(1)))*(1-c(rhor(1))))/2;
        end
    else
        %Shock x_l, x_m andx_r
        sm = (psistar+(v(rhol(end)))*(c(rhol(end))-1)...
                +(v(rhor(1)))*(1-c(rhor(1))))/2;
        if 0~=rhol(end)
            sl = [sl -f(rhol(end))/rhol(end)];
            rhol = [rhol 0];
            xl = [xl xm];
        end
        if 0~=rhor(1)
            sr = [f(rhor(1))/rhor(1) sr];
            rhor = [0 rhor];
            xr = [xm xr];
        end
    end
elseif rhol(end)>rhor(1)
    if rhor(1)==0
        if psistar<xi(0)
            %Rarefaction between um and ur
            rhom = fsolve(@(u) xi(u)-psistar, rhor(1)-epsilon,opts);
            rhom = id(rhom,epsilon);
            sm = (f(rhol(end))+f(rhom))/(rhom-rhol(end));
            w = rhom:-epsilon:rhor(1);
            if w(end)~=rhor(1)
                w = [w rhor(1)];
            end
            rhor = [w rhor(2:end)];
            sr = [(f(w(1:end-1))-f(w(2:end)))./(w(1:end-1)...
                -w(2:end)) sr];
            xr = [xm*ones(1,length(w)-1) xr];
        else
            %Shock x_l, x_m andx_r
            sm = (psistar+(v(rhol(end)))*(c(rhol(end))-1)...
                +(v(rhor(1)))*(1-c(rhor(1))))/2;
            rhol = [rhol 0];
            sl = [sl -v(rhol(end))];
            xl = [xl xm];
        end
    else
        if xi(rhor(1))>psistar
            %Rarefaction between um and ur
            rhom = fsolve(@(u) xi(u)-psistar, rhor(1)-epsilon,opts);
            rhom = id(rhom,epsilon);
            sm = (f(rhol(end))+f(rhom))/(rhom-rhol(end));
            w = rhom:-epsilon:rhor(1);
            if w(end)~=rhor(1)
                w = [w rhor(1)];
            end
            rhor = [w rhor(2:end)];
            sr = [(f(w(1:end-1))-f(w(2:end)))./(w(1:end-1)...
                -w(2:end)) sr];
            xr = [xm*ones(1,length(w)-1) xr];
        elseif xi(rhor(1))<=psistar && psistar<=theta(0)     
            rhom = fsolve(@(u) theta(u)-psistar, 0,opts);
            rhom = id(rhom,epsilon);
            sm = (f(rhol(end))+f(rhom))/(rhom-rhol(end));
            if rhom~=rhor(1)
            	sr = [(f(rhom)-f(rhor(1)))/(rhom-rhor(1)) sr];
                rhor = [rhom rhor];
                xr = [xm xr];
            end
        elseif theta(0)<psistar && psistar<lambda(0)
            %Shock x_l, x_m and x_r
            sm = (psistar+(v(rhol(end)))*(c(rhol(end))-1)...
                +(v(rhor(1)))*(1-c(rhor(1))))/2;
            rhol = [rhol 0];
            sl = [sl -v(rhol(end))];
            xl = [xl xm];
            rhor = [0 rhor];
            sr = [v(rhor(end)) sr];
            xr = [xm xr];
        else
            %Shock x_l and x_m
            rhom = fsolve(@(u) lambda(u)-psistar, 0,opts);
            rhom = id(rhom,epsilon);
            sm = (f(rhor(1))+f(rhom))/(rhor(1)-rhom);
            if rhom~=rhol(end)
                sl = [sl -(f(rhol(end))-f(rhom))/(rhol(end)-rhom)];
                rhol = [rhol rhom];
                xl = [xl xm];
            end
        end
    end
else
    if rhol(end)==0
        if psistar>xi2(0)
                %Rarefaction between ul and um
                rhom = fsolve(@(u) xi2(u)-psistar, rhol(end)-epsilon,opts);
                rhom = id(rhom,epsilon);
                sm = (f(rhor(1))+f(rhom))/(rhor(1)-rhom);
                w = rhol(end):epsilon:rhom;
                if w(end)~=rhom
                    w = [w rhom];
                end
                rhol = [rhol(1:end-1) w ];
                sl = [sl (f(w(2:end))-f(w(1:end-1)))./(w(1:end-1)...
                    -w(2:end))];
                xl = [xl xm*ones(1,length(w)-1)];
        else
                %Shock x_l, x_m and x_r
                sm = (psistar+(v(rhol(end)))*(c(rhol(end))-1)...
                    +(v(rhor(1)))*(1-c(rhor(1))))/2;
                rhor = [0 rhor];
                sr = [v(rhor(1)) sr];
                xr = [xm xr];
        end
    else
        if psistar<theta(0)
                %Shock x_m and x_r
                rhom = fsolve(@(u) theta(u)-psistar, 0,opts);
                rhom = id(rhom,epsilon);
                sm = (f(rhol(end))+f(rhom))/(rhom-rhol(end));
                if rhom~=rhor(1)
                    sr = [(f(rhom)-f(rhor(1)))/(rhom-rhor(1)) sr];
                    rhor = [rhom rhor];
                    xr = [xm xr];
                end
        elseif theta(0)<=psistar && psistar<=lambda(0)
                %Shock x_l, x_m and x_r
                sm = (psistar+(v(rhol(end)))*(c(rhol(end))-1)...
                    +(v(rhor(1)))*(1-c(rhor(1))))/2;
                rhol = [rhol 0];
                sl = [sl -v(rhol(end))];
                xl = [xl xm];
                rhor = [0 rhor];
                sr = [v(rhor(1)) sr];
                xr = [xm xr];
        elseif lambda(0)<psistar && psistar<xi2(rhol(end))
                %Shock x_l and x_m
                rhom = fsolve(@(u) lambda(u)-psistar, 0,opts);
                rhom = id(rhom,epsilon);
                sm = (f(rhor(1))+f(rhom))/(rhor(1)-rhom);
                if rhom~=rhol(end)
                    sl = [sl -(f(rhol(end))-f(rhom))/(rhol(end)-rhom)];
                    rhol = [rhol rhom];
                    xl = [xl xm];
                end
        else
                %Rarefaction between ul and um
                rhom = fsolve(@(u) xi2(u)-psistar, rhol(end)-epsilon,opts);
                rhom = id(rhom,epsilon);
                sm = (f(rhor(1))+f(rhom))/(rhor(1)-rhom);
                w = rhol(end):epsilon:rhom;
                if w(end)~=rhom
                    w = [w rhom];
                end
                rhol = [rhol(1:end-1) w ];
                sl = [sl (f(w(2:end))-f(w(1:end-1)))./(w(1:end-1)...
                    -w(2:end))];
                xl = [xl xm*ones(1,length(w)-1)];
        end
    end
end
end

                    
                    
                    
            
            
            
        
            
        
        
