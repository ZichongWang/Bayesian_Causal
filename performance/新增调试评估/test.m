%% 滤波，相当于一个卷积的操作
    QBD = imfilter(opt_QBD, fspecial('disk', 3));
    PBD = BD;
    [GTBD, GTBD_R] = readgeoraster(join([location, event, '\ground_truth\', event, '_building_damage_ground_truth_rasterized.tif']));	
    GTBD = GTBD(1:5323,1:4816);
    GTBD = double(GTBD);
    rocdet_BD = rocdetpr('BuildingDamage',QBD,QBD,GTBD,location);