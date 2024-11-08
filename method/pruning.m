%% Pruning

%% Function
function model = pruning(BD,LS,LF,sigma,side)

%% Input
% BD = {0 - without, 1 - with building}
% LS = probability [0,1], 
% LF = probability [0,1], 
% model = matrix of model classifications {1,2,3,4}
  model = zeros(size(BD,1),size(BD,2));

%% Four Local Models
if side == "single"
    % 1 - LS alone
    model = model +     ((BD == 0) & (LF <= sigma) );

    % 2 - LF alone.
    model = model +  2.*((BD == 0) & (LF > sigma) & (LF > LS));

    % 3 - LS and BD
    model = model +  3.*((BD == 1) & (LF <= sigma) );

    % 4 - LF and BD
    model = model +  4.*((BD == 1) & (LF > sigma) & (LF > LS));

else
    % 没有BD alone
    % 1 - LS alone
    % 这里BD指的是footprint
    model = model +     ((BD == 0) & (LS > LF + sigma ) & (LS > 0));

    % 2 - LF alone.
    model = model +  2.*((BD == 0) & (LF > LS + sigma ) & (LF > 0));

    % 3 - LS and BD
    model = model +  3.*((BD == 1) & (LS > LF + sigma ) & (LS > 0));

    % 4 - LF and BD
    model = model +  4.*((BD == 1) & (LF > LS + sigma ) & (LF > 0));

    % 5 - LF and LS
    model = model +  5.*((BD == 0) & (abs(LF-LS) <= sigma) );
    
    %% 注意最后没有包含& (LS > 0))或 & (LF > 0))，是故意这样做的吗？
    %% BD alone包含在这一项里面了
    % 6 - LF and LS and BD
    model = model +  6.*((BD == 1) & (abs(LF-LS) <= sigma) );
    
end
end
