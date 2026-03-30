%
% WFTCROWD v1.1
% Initial Sofware by GOATIN P. and MIMAULT M., Copyright (C) Inria (Opale team), 
% All Rights Reserved, 2012
%
function [] = WFTcrowd(x,rho,T,epsilon)

close all
tic
c = @(u) 1./(1-u);
x = [-0.543, -0.387, 0 , 0.5];
rho = [0.867, 0.839, 0.456, 0.95, 0.6];
T = 3;
epsilon = 1/250;


rho = [0 id(rho,epsilon) 0]; %Check if rho is correct
xi = turningpoint([-1 x 1],rho); %Way to determine xi
xl = [-1 x(x<xi)];  %Splitting interfaces vectors 
xr = [x(x>xi) 1];   %in two part
rhol = rho(1:length(xl)+1);     %Splitting rho
rhor = rho(end-length(xr):end); %in two part
t = 0; 
Tc = T; %Time left before end
tw = 0; %Saved time
E = 0; %Initialization of the cost integral balance error

%PdR initial
[xl xi xr rhol rhor sl si sr] = PdR(xl,xi,xr,rhol,rhor,epsilon);
xw = {[xl xi xr]}; %Saved variables
sw = {[sl si sr]};
rhow={[rhol rhor]};


figure('Units', 'pixels', 'Position', [75 75 850 375]); %Window

while t<=T
    x = [xl xi xr]; 
    s = [sl si sr];
    rho = [rhol rhor];
    rhow = {rhow{:} rho};
    collider =((x(2:end)-x(1:end-1))./(s(1:end-1)-s(2:end)));
    tc = min(collider(collider>0)); %Closest collision
    
    %cost integral balance error
    
    E = [E sum(c(rhol(2:end)).*([xl(2:end) xi]-xl))-sum(c(rhor(1:end-1)).*(xr-[xi xr(1:end-1)]))]; %#ok<AGROW>
   
    figure(1)
    if tc==Inf
        twtoc = toc; %#ok<NASGU>
        disp('tinf')
        disp('cost integral balance error')
        max(abs(E))
        xw = {xw{:} [-1 0 1]}; %#ok<NASGU>
        tw = [tw T]; %#ok<AGROW,NASGU>
        sw = {sw{:} [0 0 0]}; %#ok<NASGU>
        epsilonstr = int2str(1/epsilon);
        name = strcat('WFT',epsilonstr);
        save(name,'rhow','xw','sw','tw','twtoc','epsilon')
        return
    end
    
    if tc<=Tc
        x2 = x+s*tc;
        tp = [t t+tc t+tc t]'*ones(1,length(x)-1);
        
        subplot(1,2,1)
        hold on
        p = patch([x(1:end-1); x2(1:end-1); x2(2:end); x(2:end)],tp,'r');
        set(gca,'CLim',[0 1/epsilon])
        cdata = rho(2:end-1)/epsilon;
        set(p,'FaceColor','flat','CData',cdata,'CDataMapping','scaled')
        set(p,'EdgeColor','none')
        plot([xi; xi+si*tc],[t;t+tc],'w','LineWidth',2)
        hold off
        subplot(1,2,2) %Non-final plotting
        profil(x,rho(2:end),s,tc)
    else
        x2 = x+s*Tc;
        tp = [t T T t]'*ones(1,length(x)-1);
        subplot(1,2,1)
        hold on
        p = patch([x(1:end-1); x2(1:end-1); x2(2:end); x(2:end)],tp,'r');
        set(gca,'CLim',[0 1/epsilon])
        cdata = rho(2:end-1)/epsilon;
        set(p,'FaceColor','flat','CData',cdata,'CDataMapping','scaled')
        set(p,'EdgeColor','none')
        plot([xi; xi+si*Tc],[t;T],'w','LineWidth',2);
        hold off
        subplot(1,2,2) %Final plotting
        profil(x,rho(2:end),s,tc)
    end
    
    
t = t+tc;
tw = [tw t]; %#ok<AGROW>
Tc = Tc-tc;
xl = tc*sl+xl;
xi = tc*si+xi; %Main update
xr = tc*sr+xr;



xf = [xl xi];
rhof = [rhol rhor(1)];
rhof(find(abs(xf(1:end-1)-xf(2:end))<10^-14)+1)=[];   
xf(find(abs(xf(1:end-1)-xf(2:end))<10^-14)+1)=[];  
xl = xf(1:end-1);
xi = xf(end); %Suppress redundances, 0 and Nan
rhol = rhof(1:end-1);

xf = [xi xr];
rhof = [rhol(end) rhor];
rhof(find(abs(xf(1:end-1)-xf(2:end))<10^-14)+1)=[];
xf(find(abs(xf(1:end-1)-xf(2:end))<10^-14)+1)=[];
xr = xf(2:end);
xi = xf(1); %Suppress redundances, 0 and Nan
rhor = rhof(2:end);

[xl xi xr rhol rhor sl si sr] = PdR(xl,xi,xr,rhol,rhor,epsilon);
xw = {xw{:} [xl xi xr]};
sw = {sw{:} [sl si sr]};

end
twtoc = toc; %#ok<NASGU>
xw = {xw{:} [-1 0 1]}; %#ok<NASGU>
disp('cost integral balance error')
max(abs(E))
epsilonstr = int2str(1/epsilon);
name = strcat('WFT',epsilonstr);
save(name,'rhow','xw','sw','tw','twtoc','epsilon')
end
