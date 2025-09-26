
function [final_t, len_all_stored_data, all_stored_data, success] = WFTcrowd_1(x, rho, x_vals, rho_vals, T, epsilon)
close all
tic


c = @(u) 1./(1-u);

rho = [0 id(rho,epsilon) 0]; % give boundary value of rho = 0 and computes rho in between, after step 1 = [0,0.1,0.9,0]
xi = turningpoint([-1 x 1],rho); % find xi also assign xi based on rho, after step 1: computed xi = 0.444
xl = [-1 x(x<xi)];  % LHS pos , after step 1 = [-1,0]
xr = [x(x>xi) 1];    % RHS pos, after step 1 = 1
rhol = rho(1:length(xl)+1);     % LHS rho after step 1 = [0,0.1,0.9]
rhor = rho(end-length(xr):end); %RHS rho, after step 1 = [0.9, 0]

t = 0;
Tc = T; %Time left before end
tw = 0; %Saved time
E = 0; %Initialization of the cost integral balance error

%PdR initial
[xl xi xr rhol rhor sl si sr] = PdR(xl,xi,xr,rhol,rhor,epsilon);

xw = {[xl xi xr]}; % after step 1: xl = [-1,0,0,...201] xr = [0.44, 1,1,...102] xi = 0.44
sw = {[sl si sr]}; % after step 1: sl = [0, -0.796 ...0.796 201] 101 negative then 100 pos (-0.79 to 0.79 equidistant)  
rhow={[rhol rhor]}; % after step 1 : shape = vector 305 elements. 

% Initialize an empty cell array to store stored_data from each iteration
all_stored_data = {}; % This will hold stored_data for all iterations
all_stored_data = {x_vals,rho_vals, round(xi,2), round(xi,2) };

%figure('Units', 'pixels', 'Position', [75 75 850 375]); %Window
last_time_saved = 0;

while t<=T
    success = true;
    x = [xl xi xr]; 
    s = [sl si sr];
    rho = [rhol rhor];
    rhow = {rhow{:} rho}; % after step 1 : rhow = [rhow rho] two cells 1x305 1x305
    collider =((x(2:end)-x(1:end-1))./(s(1:end-1)-s(2:end))); % after step 1 = [1.25,0,0,...303]
    tc = min(collider(collider>0)); %Closest collision
    
    %cost integral balance error
    
    E = [E sum(c(rhol(2:end)).*([xl(2:end) xi]-xl))-sum(c(rhor(1:end-1)).*(xr-[xi xr(1:end-1)]))]; %#ok<AGROW> after step1: [0,0]
   
    %figure(1)
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

        xi_target = xi+si*tc;
        [stored_data , last_time_saved] = profil(x, rho(2:end), s, tc,xi,xi_target, t, last_time_saved);  % Pass the storage variable to profil
        
    else

        xi_target = xi+si*tc;
        [stored_data , last_time_saved] = profil(x, rho(2:end), s, tc,xi,xi_target, t, last_time_saved);
        
    end


% Append stored_data to all_stored_data
if ~all(cellfun(@isempty, stored_data(:)))
    % Append non-empty cell data to all_stored_data
    all_stored_data = [all_stored_data; stored_data];
end


real = cell2mat(all_stored_data(:, 3)); % Column 3
computed = cell2mat(all_stored_data(:, 4)); % Column 4

% Compute the sum of differences
sum_of_differences = abs(real - computed);






t = t+tc;
tw = [tw t]; %#ok<AGROW>
Tc = Tc-tc;
xl = tc*sl+xl;
xi = tc*si+xi; %Main update
xr = tc*sr+xr;


% these two bolcks are used to remove near zero differnces in values 
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


final_t = t; % The final value of t
len_all_stored_data = length(all_stored_data); % The length of all_stored_data


if any(sum_of_differences > 0.051)
    success = false;
    disp('high error in tp, Exiting the function.');
    return;  % Exit the function early
end


end

min_allow_tp_range = 0.3; % minimum allowed range
max_allow_tp_range = 1; % maximum allowed range

turningpoints_all = cell2mat(all_stored_data(:, 3));
turningpoint_difference = abs(max(turningpoints_all) - min(turningpoints_all));


if turningpoint_difference > max_allow_tp_range || turningpoint_difference < min_allow_tp_range
    disp('Range of turning points is out of acceptable bounds (0.3 to 1), Exiting the function.');
    success = false;
    return; 
end

%fprintf('The value of the variable is: %f\n', k)
twtoc = toc; %#ok<NASGU>
xw = {xw{:} [-1 0 1]}; %#ok<NASGU>
%disp('cost integral balance error')
%max(abs(E))
epsilonstr = int2str(1/epsilon);
name = strcat('WFT',epsilonstr);
save(name,'rhow','xw','sw','tw','twtoc','epsilon')

end
