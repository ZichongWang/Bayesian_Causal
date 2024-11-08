%% Performance

% TPR - true positive rate
% FPR - false positive rate
% TNR - true negative rate
% FNR - false negative rate
% CEL - cross-entropy loss
% ROC - receiver operating characteristics curve
% DET - detection error trade-off curve
%% 不要关闭已经输出的图片窗口！！
%% 不要关闭已经输出的图片窗口！！
%% 不要关闭已经输出的图片窗口！！
%% Initialize
close all; clear; clc
tic
%% Import
location = 'C:\Users\Administrator\Desktop\大创文件\data\';
event = '2023_turkey';
filename = join([location, '2023_turkeylambda0_sigma0_prunedouble.mat']);
load(filename)                                                          % contains prior and posterior estimates

%% Declare variables

    % Landslide
    PLS = LS;
%     PLS = tmp_LS;
%% 滤波，相当于一个卷积的操作
    QLS = imfilter(final_QLS, fspecial('disk', 3));
    [GTLS, GTLS_R] = readgeoraster(join([location, event, '\ground_truth\', event, '_landslide_ground_truth_rasterized.tif']));	% landslide groundtruth

    % Liquefaction
    PLF = LF;
%     PLF = tmp_LF;
%% 滤波，相当于一个卷积的操作
%     QLF = imfilter(final_QLF, fspecial('average', [3 3]));
    QLF = imfilter(final_QLF, fspecial('disk', 20));
    [GTLF, GTLF_R] = readgeoraster(join([location, event, '\ground_truth\', event, '_liquefaction_ground_truth_rasterized.tif']));   % liquefaction groundtruth

% %% Compute TPR, FPR, TNR, FNR
%     
%     prio_thresh = 0;
%     post_thresh = 0;
%     
%     % Landslide
%     [PLS_TPR,PLS_FPR,PLS_TNR,PLS_FNR,   ...
%      QLS_TPR,QLS_FPR,QLS_TNR,QLS_FNR]   ...
%     = binaryerror(PLS,QLS,GTLS,prio_thresh,post_thresh);
% 
%     % Liquefaction
%     [PLF_TPR,PLF_FPR,PLF_TNR,PLF_FNR,   ...
%      QLF_TPR,QLF_FPR,QLF_TNR,QLF_FNR]   ...
%     = binaryerror(PLF,QLF,GTLF,prio_thresh,post_thresh);
% 
% %% Compute CEL
%     
%     % Landslide
%     [ploss_LS, qloss_LS] = cel(PLS,QLS,GTLS);
% 
%     % Liquefaction
%     [ploss_LF, qloss_LF] = cel(PLF,QLF,GTLF);

%% 复数转化
QLS = real(QLS);
QLF = real(QLF);
%% Generate ROC, DET

    % Landslide
    rocdet_LS = rocdetpr('Landslide',PLS,QLS,GTLS,location);
    
    % Liquefaction
    rocdet_LF = rocdetpr('Liquefaction',PLF,QLF,GTLF,location);

%% Save File
% filename=join([location,'performance.mat']);
% save(filename);


%     我们自己写的
%     building damage

%% 滤波，相当于一个卷积的操作
    QBD = imfilter(opt_QBD, fspecial('disk', 3));
    [GTBD, GTBD_R] = readgeoraster(join([location, event, '/ground_truth/', event, '_building_damage_ground_truth_rasterized.tif']));	% landslide groundtruth
    rocdet_BD = rocdetpr('BuildingDamage',QBD,QBD,GTBD,location);
    



