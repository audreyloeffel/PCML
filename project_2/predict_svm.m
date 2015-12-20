load('data/test.mat');
load('data/train.mat');

% WORK ! :)

yTrBin = train.y;
yTrBin(yTrBin~=4) = 1;
yTrBin(yTrBin==4) = 0;

XTr = train.X_hog;
options = statset('UseParallel',1);
t = templateSVM('BoxConstraint',0.11,'KernelFunction','linear');
LinAllModel = fitcecoc(XTr,yTrBin,'Learners',t,'Coding','onevsall','Options',options);
yTr_hat = predict(LinAllModel,XTr);

errBinSVM2 = ber(yTrBin, yTr_hat);