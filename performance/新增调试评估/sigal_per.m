close all; clear; clc
tic
%% Import
location = 'C:\Users\Administrator\Desktop\大创文件\data\';
event = '2024_japan2';
filename = join([location, '2024_japan2lambda0_sigma0_prunedouble.mat']);
load(filename)                                                          % contains prior and posterior estimates


   % Landslide
    PLS = LS;
%     PLS = tmp_LS;
%% 滤波，相当于一个卷积的操作
    QLS = imfilter(final_QLS, fspecial('disk', 3));
    [GTLS, GTLS_R] = readgeoraster(join([location, event, '\ground_truth\', event, '_landslide_ground_truth_rasterized.tif']));	% landslide groundtruth
QLS = real(QLS);

%% Generate ROC, DET

    % Landslide
    rocdet_LS = rocdetpr('Landslide',PLS,QLS,GTLS,location);
