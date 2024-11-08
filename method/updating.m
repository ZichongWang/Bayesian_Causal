%% Updating
cd('C:\Users\Administrator\Desktop\大创文件')
%% Initialize
close all; clear; clc
tic
%% Import
%% 导入tif数据，并且转化为特定格式，前一个变量是矩阵，可以直接可视化，与论文上的图是一致的。imagesc(LF)
%% ../ 是返回上层路径，需要返回两层，然后新增一个event变量，通过更改其值（数据集名称），即可访问不同数据集
%% 用imagesc()可以初步可视化
%% 可以看出LS和LF的粒度非常粗，与论文中的结果是一致的。观察其矩阵也是有一个局部相等的结构，这是因为先验给的观测其实是非常粗略的。
%% 但是这里却选用了与DPM粒度相同的图，因此相当于把粗的像素格子切成细的小块，每个小块的值与原先的大格子是相同的。
%%location = '../../data/';
location = 'C:\Users\Administrator\Desktop\大创文件\data\';
event = '2024_japan2';
%% 这两行不太确定传得对不对，通读一遍应该会更理解一点。
[Y, Y_R] = readgeoraster(join([location, event,'\damage_proxy_map\', event, '_damage_proxy_map.tif']));
[BD, BD_R] = readgeoraster(join([location, event,'\building_footprint\', event, '_building_footprint_rasterized.tif']));

[LS, LS_R] = readgeoraster(join([location, event, '\prior_models\', event, '_prior_landslide_model.tif']));
[LF, LF_R] = readgeoraster(join([location, event,'\prior_models\', event, '_prior_liquefaction_model.tif']));


%% Fix Data Input
%% 看起来是要把footprint数据二值化，那这样做其实是会丢失一些信息的，后续留意一下。
%% nan值直接置为0，后续留意一下nan值是如何产生的。
BD(BD>0) = 1; 
Y(isnan(Y))=0;
BD(isnan(BD))=0;
LS(isnan(LS))=0;
LF(isnan(LF))=0;
%% clip
%Y = Y(1:5323,1:4816);
%BD = BD(1:5323,1:4816);
%Y = Y(1:2616,1:2837);
%Y = double(Y);
%BD = BD(1:845,1:1558);
%BD = double(BD);
%LS = LS(1:2616,1:2837);
%LF = LF(1:2616,1:2837);
Y = (Y+10)/20;
%% Convert Landslide Areal Percentages to Probabilities
%% LS数据是已经经过处理的数据，叫做LAP数据，最原始的数据应该是点类型和多边形类型的混合
%% 下面的LS数据也是一样的。
%% 基于多边形数据计算LAP比较容易，基于点数据可能还得插值一下，插值完还需要判断landslide的0和1，这可能需要一些领域知识
%% 那sampling是干嘛的？决定去“勘测”哪些点吗？还是说这些已经观测到了，只是统计上的sampling？
%% 还有一个问题是这里会解出复根，只取实部是否合理？或者怎么解释这件事？
new_LS = LS;
index = find(LS>0);
for i = index'
    p = [4.035 -3.042 5.237 (-7.592-log(LS(i)))];
    tmp_root = roots(p);
    new_LS(i) = real(tmp_root(imag(tmp_root)==0));
end
disp('Converted Landslide Areal Percentages to Probabilities')
toc

%% Convert Liquefaction Areal Percentages to Probabilities
%% 选用了模型2的参数
%% 读一下模型1和2的区别，为什么选用模型2的参数。
%% 所谓模型其实是用来拟合这个概率数据和液化/滑坡百分比的曲线，这个概率数据是怎么来的呢？
%% 是基于前人的一些工作，其实概率数据本来就是从某些地方预测出来的，但是大家用的多了之后也许当成了一个行业标准，就成为真值了（对于某些历史数据）
%% 就可以用那些历史数据去拟合一个曲线出来，虽然这个曲线看起来似乎也很没有道理，但是如果业内的大家都承认的话，maybe not bad
%% 为什么参数a移动了两个小数点？49.15改成了0.4915

%% 先验怎么代入？
new_LF = LF;
index = find(LF>0);
for i = index'
    new_LF(i) = (log((sqrt(0.4915./LF(i)) - 1)./42.40))./(-9.165);
% new_LF(i) = (log((sqrt(49.15./LF(i)) - 1)./42.40))./(-9.165);
end
disp('Converted Liquefaction Areal Percentages to Probabilities')
toc

%% Change into Non-negative Probabilities
%% 其实也是很没有道理的，他的拟合曲线严格来看其实是不合理的，没有“外推”的能力。会出现0值和nan值。
new_LF(new_LF<0) = 0;
new_LS(new_LS<0) = 0;
new_LS(isnan(new_LS)) = 0;
new_LF(isnan(new_LF)) = 0;
tmp_LF = new_LF;
tmp_LS = new_LS;

%% Classify Local Model by Pruning
prune_type = 'double';
sigma = 0;
%% 取绝对值对应的意思是，一个高概率发生的话，另一个不太可能发生。论文中只提到了其中一边。
%% sigma相当于是一个差异的阈值，高于这个阈值就认为不发生了，这样的话因为这个阈值是中位数，所以LS和LF加起来会砍掉一半
%sigma = median(abs(new_LS((LS>0)&(LF>0)) - new_LF((LS>0)&(LF>0)))); 
% sigma = zeros(2616, 2837);
%% 用1-6标记了LS、LF、BD的组合类型
%% 这把LS=0且LF=0与LS=1且LF=1这样的情形划在了一起
%% 也就是说只要这俩概率很接近就直接划在一起，因为一个发生另一个就不太可能发生。


LOCAL = pruning(BD,tmp_LS,tmp_LF, sigma, prune_type);
%% 两个min返回矩阵的最小值
%% 这里相当于给那两种类型都赋予一个小的激活，不管之前是全0还是全1，因为之前是将两件事情独立考虑的
%% 现在的模型是要联合起来考虑
tmp_LS ((LOCAL==5)|(LOCAL==6)) = min(min(new_LS(new_LS>0)));
tmp_LF ((LOCAL==5)|(LOCAL==6)) = min(min(new_LF(new_LF>0)));

%% Set Lambda Term
% lambda = 0;
%% 调整
lambda = 0;

%% Initialize Weight Vector w    
% [w0;weps;   w0BD;w0LS;w0LF;   wLSBD;wLFBD;   wBDy;wLSy;wLFy;   weLS;weLF;weBD;  waLS;waLF]
%% 为什么负？为什么0？
w = rand([15,1]);
w([4,5]) = 0;
w([1,3]) = -1.*w([1,3]);
%% 调整
% regu_type = 1;  
regu_type = 2;  

%% Set Variational hyperparameters
% Nq = 10;            % Number of Posterior Probability Iterations
%% 调整
Nq = 10;  

%% Set Weight Updating Parameters
rho = 1e-3;         % Step size
delta = 1e-3;   	% Acceptable tolerance for weight optimization
eps_0 = 0.001;   	% Lower-bound non-negative weight

% k_divide = 5;  %% 将5000 * 5600 分成 k * k
% nrow = 5000/k_divide;
% ncol = 5600/k_divide;
% 
% Y_reshape = mat2cell(Y,[5000 71], [5600 42]);
% [Y_1, Y_2, Y_3, Y_4] = deal(Y_reshape{:});
% Y_Y = [];
% for i = 1:k_divide
%     for j = 1:k_divide
%         Y_Y(:,:,(k_divide*(i-1)+j)) = Y_1((nrow*(i-1)+1):(nrow*(i)),(ncol*(j-1)+1):(ncol*(j)));
%     end
% end
% 
% tmp_LS_reshape = mat2cell(tmp_LS,[5000 71], [5600 42]);
% [tmp_LS_1, tmp_LS_2, tmp_LS_3, tmp_LS_4] = deal(tmp_LS_reshape{:});
% tmp_LS_tmp_LS = [];
% for i = 1:k_divide
%     for j = 1:k_divide
%         tmp_LS_tmp_LS(:,:,(k_divide*(i-1)+j)) = tmp_LS_1((nrow*(i-1)+1):(nrow*(i)),(ncol*(j-1)+1):(ncol*(j)));
%     end
% end
% 
% tmp_LF_reshape = mat2cell(tmp_LF,[5000 71], [5600 42]);
% [tmp_LF_1, tmp_LF_2, tmp_LF_3, tmp_LF_4] = deal(tmp_LF_reshape{:});
% tmp_LF_tmp_LF = [];
% for i = 1:k_divide
%     for j = 1:k_divide
%         tmp_LF_tmp_LF(:,:,(k_divide*(i-1)+j)) = tmp_LF_1((nrow*(i-1)+1):(nrow*(i)),(ncol*(j-1)+1):(ncol*(j)));
%     end
% end
% 
% LOCAL_reshape = mat2cell(LOCAL,[5000 71], [5600 42]);
% [LOCAL_1, LOCAL_2, LOCAL_3, LOCAL_4] = deal(LOCAL_reshape{:});
% LOCAL_LOCAL = [];
% for i = 1:k_divide
%     for j = 1:k_divide
%         LOCAL_LOCAL(:,:,(k_divide*(i-1)+j)) = LOCAL_1((nrow*(i-1)+1):(nrow*(i)),(ncol*(j-1)+1):(ncol*(j)));
%     end
% end

% [opt_w, opt_QBD, opt_QLS, opt_QLF, QLS, QLF, QBD, final_loss, best_loss, local] = ...
%     SVI(Y_Y(:,:,1),tmp_LS_tmp_LS(:,:,1),tmp_LF_tmp_LF(:,:,1),w,Nq,rho,delta,eps_0,LOCAL_LOCAL(:,:,1),lambda,regu_type,sigma,prune_type);

%% Output
%% 传入的参数没有building footprint，因为这个变量的信息已经被“编码”到其他LS和LF中了（0-6）
%% 影响了每个点处相关的因果图的形状，可以理解成已经按照building footprint这个变量进行分层了，
%% 在每一层中这个变量已经不见了。
[opt_w, opt_QBD, opt_QLS, opt_QLF, QLS, QLF, QBD, final_loss, best_loss, local] = ...
    SVI(Y,tmp_LS,tmp_LF,w,Nq,rho,delta,eps_0,LOCAL,lambda,regu_type,sigma,prune_type);


%% Convert Probabilities to Areal Percentages
final_QLS = exp(-7.592 + 5.237.*opt_QLS - 3.042*opt_QLS.*opt_QLS + 4.035.*opt_QLS.*opt_QLS.*opt_QLS);
%% 这里用的还是0.4915，而不是49.15，论文中有指示要乘一个0.01吗
final_QLF = 0.4915./(1+42.40 .* exp(-9.165.*opt_QLF)).^2;
% final_QLF = 49.15./(1+42.40 .* exp(-9.165.*opt_QLF)).^2;

%% Rounddown Very Small Areal Percentages to Zero
%% 概率为0的强制设置areal percentage为0，可以理解成作者写了一个分段的函数作为model，把那些无意义的点截断了。
final_QLS(final_QLS<=exp(-7.592)) = 0;
final_QLF(final_QLF<=0.4915./(1+42.40).^2) = 0;
% final_QLF(final_QLF<=49.15./(1+42.40).^2) = 0;

%% Remove probabilities in water bodies
%% 若先验都为0，则不可能发生。
final_QLS((LS==0)&(LF==0))=0;
final_QLF((LS==0)&(LF==0))=0;

%% Export GeoTIFF Files
geotiffwrite('QLS.tif', final_QLS, LS_R)
geotiffwrite('QLF.tif', final_QLF, LS_R)
geotiffwrite('QBD.tif', opt_QBD, LS_R)

%% Export All to a File
filename=join([location, event,'lambda',num2str(lambda), '_sigma',num2str(sigma),'_prune',prune_type,'.mat']);
save(filename);