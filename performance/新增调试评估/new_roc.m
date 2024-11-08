close all; clear; clc
tic
%% Import
location = 'C:\Users\Administrator\Desktop\大创文件\data\';
event = '2024_japan2';
filename = join([location, '2024_japan2lambda0_sigma0_prunedouble.mat']);
load(filename)                                                          % contains prior and posterior estimates

%% Declare variables

    % Landslide
    PLS = LS;
%     PLS = tmp_LS;
%% 滤波，相当于一个卷积的操作
    QLS = imfilter(final_QLS, fspecial('disk', 3));
    [GTLS, GTLS_R] = readgeoraster(join([location, event, '\ground_truth\', event, '_landslide_ground_truth_rasterized.tif']));	% landslide groundtruth
    %QLS = real(QLS);

    % Landslide
    new_cal_roc("Landslide",PLS,QLS,GTLS,location);

     % Liquefaction
    PLF = LF;
    QLF = imfilter(final_QLF, fspecial('disk', 2));
    [GTLF, GTLF_R] = readgeoraster(join([location, event, '\ground_truth\', event, '_liquefaction_ground_truth_rasterized.tif']));   % liquefaction groundtruth
    QLF = real(QLF);
   
    new_cal_roc('Liquefaction',PLF,QLF,GTLF,location);

