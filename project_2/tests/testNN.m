%
% Test NN with Hog and CNN features for the binary an multiclass
% classification.
%

load('data/train.mat');
load('data/pca_Xcnn.mat');
load('data/pca_Xhog.mat');

yTr = double(train.y);
XTrcnn = Xcnn;
XTrhog = Xhog;
Xtrall = [Xcnn Xhog];

K = 5;
numepochs = 30;
batchsize = 50;
rate = 2;
neuralFt = 10;

fprintf('---------- BINARY ------------);
% Binary classification : 1 = horse car airpane, 0 = others
yTrBin = yTr;
yTrBin(yTrBin~=4) = 1;
yTrBin(yTrBin==4) = 0;

[cnnberTr, cnnberTe, cnnberTr2, cnnberTe2] = crossValidationNN(XTrcnn, yTrBin, K, neuralFt, numepochs, batchsize, rate);
cBTr = mean(cnnberTr);
cBTe = mean(cnnberTe);
cBTr2 = mean(cnnberTr2);
cBte2 = mean(cnnberTe2);

[hogberTr, hogberTe, hogberTr2, hogberTe2] = crossValidationNN(XTrcnn, yTrBin, K, neuralFt, numepochs, batchsize, rate);
hBTr = mean(hogberTr);
hBTe = mean(hogberTe);
hBTr2 = mean(hogberTr2);
hBte2 = mean(hogberTe2);


[aberTr, aberTe, aberTr2, aberTe2] = crossValidationNN(XTrcnn, yTrBin, K, neuralFt, numepochs, batchsize, rate);
aBTr = mean(aberTr);
aBTe = mean(aberTe);
aBTr2 = mean(aberTr2);
aBte2 = mean(aberTe2);

fprintf('---------- MULTICLASS ------------);

[cmberTr, cmbernTe, cmberTr2, cmberTe2] = crossValidationNN(XTrcnn, yTr, K, neuralFt, numepochs, batchsize, rate);
cMTr = mean(cmberTr);
cMTe = mean(cmbernTe);
cMTr2 = mean(cmberTr2);
cMte2 = mean(cmberTe2);

[hmberTr, hmberTe, hmberTr2, hmberTe2] = crossValidationNN(XTrcnn, yTr, K, neuralFt, numepochs, batchsize, rate);
hMTr = mean(hmberTr);
hMBTe = mean(hmberTe);
hMTr2 = mean(hmberTr2);
hMte2 = mean(hmberTe2);
save('data/wsTestSVM.mat');



[amerTr, amerTe, amerTr2, amerTe2] = crossValidationNN(XTrcnn, yTr, K, neuralFt, numepochs, batchsize, rate);
aMTr = mean(amerTr);
aMTe = mean(amerTe);
aMTr2 = mean(amerTr2);
aMte2 = mean(amerTe2);

save('data/wsTestSVM.mat');