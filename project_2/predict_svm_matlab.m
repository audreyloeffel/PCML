load('data/test.mat');
load('data/train.mat');

% DOESN'T WORK - do not converge ...

yTrBin = train.y;
yTrBin(yTrBin~=4) = 1;
yTrBin(yTrBin==4) = 0;

model = svmtrain(train.X_hog, yTrBin, 'boxconstraint', 0.1);
predBin = svmclassify(model, test.X_hog);

predMulti = multisvm(train.X_hog, train.y, test.X_hog, 'boxconstraint', 0.1);

