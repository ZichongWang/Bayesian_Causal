info = geotiffinfo(join([location, event, '\prior_models\', event, '_prior_landslide_model.tif']));
info.SpatialRef.RasterSize = [2616, 2837];
LS_R_clip = info.SpatialRef;

geotiffwrite('QLS.tif', final_QLS, LS_R_clip)
geotiffwrite('QLF.tif', final_QLF, LS_R_clip)
geotiffwrite('QBD.tif', opt_QBD, LS_R_clip)

filename=join([location, event,'lambda',num2str(lambda), '_sigma',num2str(sigma),'_prune',prune_type,'.mat']);
save(filename);