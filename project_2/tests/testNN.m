%
% Test NN with Hog and CNN features for the binary an multiclass
% classification.
%

load('data/train.mat');
load('data/X_cnnSVD.mat');
load('data/X_hogSVD.mat');

yTr = double(train.y);
XTrcnn = normalizeMe(Xcnn);
XTrhog = normalizeMe(Xhog);
Xtrall = [XTrcnn XTrhog];

K = 4;
numepochs = 20;
batchsize = 100;
rate = 2;
neuralFt = 10;

fprintf('---------- BINARY ------------');
% Binary classification : 1 = horse car airpane, 0 = others
yTrBin = yTr;
yTrBin(yTrBin~=4) = 1;
yTrBin(yTrBin==4) = 0;

[cnnberTr, cnnberTe] = crossValidationNN(XTrcnn, yTrBin, K, neuralFt, numepochs, batchsize, rate);
cBTr = mean(cnnberTr);
cBTe = mean(cnnberTe);
save('wsTestNNm.mat');
[hogberTr, hogberTe] = crossValidationNN(XTrcnn, yTrBin, K, neuralFt, numepochs, batchsize, rate);
hBTr = mean(hogberTr);
hBTe = mean(hogberTe);
save('wsTestNNm.mat');

[aberTr, aberTe] = crossValidationNN(XTrcnn, yTrBin, K, neuralFt, numepochs, batchsize, rate);
aBTr = mean(aberTr);
aBTe = mean(aberTe);
save('wsTestNNm.mat');
fprintf('---------- MULTICLASS ------------');

[cmberTr, cmbernTe] = crossValidationNN(XTrcnn, yTr, K, neuralFt, numepochs, batchsize, rate);
cMTr = mean(cmberTr);
cMTe = mean(cmbernTe);
save('wsTestNNm.mat');
[hmberTr, hmberTe] = crossValidationNN(XTrcnn, yTr, K, neuralFt, numepochs, batchsize, rate);
hMTr = mean(hmberTr);
hMBTe = mean(hmberTe);
save('wsTestNNm.mat');

[amerTr, amerTe] = crossValidationNN(XTrcnn, yTr, K, neuralFt, numepochs, batchsize, rate);
aMTr = mean(amerTr);
aMTe = mean(amerTe);

save('wsTestNNm.mat');