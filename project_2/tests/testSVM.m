%
% Test SVM with Hog and CNN features for the binary an multiclass
% classification.
%

% load('data/train.mat');
% load('data/pca_Xcnn.mat');
% load('data/pca_Xhog.mat');

yTr = double(train.y);
XTrcnn = normalizeMe(Xcnn);
XTrhog = normalizeMe(Xhog);
XTrall = [Xcnn Xhog];

K = 4;


fprintf('CNN Binary SVM\n');
[cnnberTr, cnnberTe] = crossValidation(XTrcnn, yTr, K, 'binSVM');
cBTr = mean(cnnberTr);
cBTe = mean(cnnberTe);
fprintf('HOG Binary SVM\n');
[hogberTr, hogberTe] = crossValidation(XTrhog, yTr, K,'binSVM');
hBTr = mean(hogberTr);
hBTe = mean(hogberTe);
fprintf('CNN Multi SVM\n');
[cmberTr, cmbernTe] = crossValidation(XTrcnn, yTr, K, 'multiSVM');
cMTr = mean(cmberTr);
cMTe = mean(cmbernTe);
fprintf('HOG Multi SVM\n');
[hmberTr, hmberTe,] = crossValidation(XTrcnn, yTr, K, 'multiSVM');
hMTr = mean(hmberTr);
hMBTe = mean(hmberTe);
save('data/wsTestSVM.mat');
%%
fprintf('BOTH Binary SVM\n');
[aberTr, aberTe] = crossValidation(XTrall, yTr, K, 'binSVM');
aBTr = mean(aberTr);
aBTe = mean(aberTe);
fprintf('BOTH Multi SVM\n');
[amerTr, amerTe] = crossValidation(XTrall, yTr, K, 'multiSVM');
aMTr = mean(amerTr);
aMTe = mean(amerTe);

save('data/wsTestSVM.mat');
