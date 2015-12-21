load('data/test.mat');
load('data/train.mat');
addpath(genpath('../toolbox/DeepLearnToolbox-master'));

%% APPLYING PCA

% -> saved in files for better running time.

% Xnorm = double(train.X_cnn);
% Xnorm(:, end) = [];
% k = 500;
% 
% [U, mu, vars] = pca(Xnorm);
% [yk, Xcnn, truc] = pcaApply(Xnorm, U, mu, k);
% save('data/pca_Xcnn.mat', 'Xcnn');
% 
% Xnorm = double(train.X_hog);
% Xnorm(:, end) = [];
% k = 400;
% 
% [U, mu, vars] = pca(Xnorm);
% [yk, Xhog, truc] = pcaApply(Xnorm, U, mu, k);
% save('data/pca_Xhog.mat', 'Xhog');

load('data/pca_Xcnn.mat');
load('data/pca_Xhog.mat');

%% NORMALIZYING AND FORMATING DATA

yTr = double(train.y);
XTrcnn = normalizeMe(Xcnn);
XTrhog = normalizeMe(Xhog);
XTrall = [Xcnn Xhog];
XTecnn = test.X_cnn;
XTehog = test.X_hog;

%% BINARY CLASSIFICATION

yTrBin = train.y;
yTrBin(yTrBin~=4) = 1; % cars, horses and airplanes
yTrBin(yTrBin==4) = 0; % other objects

XTr = train.X_cnn;
options = statset('UseParallel',1);
t = templateSVM('BoxConstraint',0.11,'KernelFunction','linear');
LinAllModel = fitcecoc(XTr,yTrBin,'Learners',t,'Coding','onevsall','Options',options);
predMulti = predict(LinAllModel,XTr);
Ytest = predict(LinAllModel,XTecnn);

%save('pred_binary.mat', 'Ytest');

%% MULTICLASS CLASSIFICATION

XTr = train.X_cnn;
options = statset('UseParallel',1);
t = templateSVM('BoxConstraint',0.11,'KernelFunction','linear');
LinAllModel = fitcecoc(XTr,yTr,'Learners',t,'Coding','onevsall','Options',options);
predMulti = predict(LinAllModel,XTr);
Ytest = predict(LinAllModel,XTecnn);

%save('pred_multiclass.mat', 'Ytest');