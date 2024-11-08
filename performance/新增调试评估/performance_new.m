close all; clear; clc
tic
%% Import
location = 'C:\Users\Administrator\Desktop\大创文件\data\';
event = '2024_japan2';
filename = join([location, '2024_japan2lambda0_sigma0_prunedouble.mat']);
load(filename)                                                          % contains prior and posterior estimates

QBD = imfilter(opt_QBD, fspecial('disk', 1));
QBD = real(QBD);
    [GTBD, GTBD_R] = readgeoraster(join([location, event, '/ground_truth/', event, '_building_damage_ground_truth_rasterized.tif']));	% landslide groundtruth
    rocdet_BD = rocdetpr('BuildingDamage',QBD,QBD,GTBD,location);
%% Declare variables

    % Landslide
    PLS = LS;
%     PLS = tmp_LS;
%% 滤波，相当于一个卷积的操作
    QLS = imfilter(final_QLS, fspecial('disk', 3));
    [GTLS, GTLS_R] = readgeoraster(join([location, event, '\ground_truth\', event, '_landslide_ground_truth_rasterized.tif']));	% landslide groundtruth

    
%% 复数转化
QLS = real(QLS);

%% Generate ROC, DET

    % Landslide
    evaluation_map('Landslide',PLS,QLS,GTLS,location);

% Liquefaction
    PLF = LF;
%     PLF = tmp_LF;
%% 滤波，相当于一个卷积的操作
%     QLF = imfilter(final_QLF, fspecial('average', [3 3]));
    QLF = imfilter(final_QLF, fspecial('disk', 20));
    [GTLF, GTLF_R] = readgeoraster(join([location, event, '\ground_truth\', event, '_liquefaction_ground_truth_rasterized.tif']));   % liquefaction groundtruth


    QLF = real(QLF);
    % Liquefaction
    evaluation_map('Liquefaction',PLF,QLF,GTLF,location);
    
    