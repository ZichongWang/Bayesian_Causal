close all; clear; clc
tic
%% Import
location = 'C:\Users\Administrator\Desktop\大创文件\data\';
event = '2023_turkey';
filename = join([location, '2023_turkeylambda0_sigma0_prunedouble_bd.mat']);
load(filename)                                                          % contains prior and posterior estimates

QBD = imfilter(opt_QBD, fspecial('disk', 3));
    [GTBD, GTBD_R] = readgeoraster(join([location, event, '/ground_truth/', event, '_building_damage_ground_truth_rasterized.tif']));	% landslide groundtruth
    QBD = QBD(1:5323,1:4816);
    GTBD = GTBD(1:5323,1:4816);
    evaluation_map('BuildingDamage',QBD,QBD,GTBD,location);

