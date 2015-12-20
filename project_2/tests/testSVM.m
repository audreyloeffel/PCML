%
% Test SVM with Hog and CNN features for the binary an multiclass
% classification.
%

load('data/train.mat');
load('data/pca_Xcnn.mat');
load('data/pca_Xhog.mat');

yTr = double(train.y);
XTrcnn = Xcnn;
XTrhog = Xhog;
Xtrall = [Xcnn Xhog];
gamma = 1;
C = 1;
[cnnberTr, cnnberTe, cnnberTr2, cnnberTe2] = crossValidation(XTrcnn, yTr, 5, C, gamma, 'binSVM');
cBTr = mean(cnnberTr);
cBTe = mean(cnnberTe);
cBTr2 = mean(cnnberTr2);
cBte2 = mean(cnnberTe2);

[hogberTr, hogberTe, hogberTr2, hogberTe2] = crossValidation(XTrhog, yTr, 5, C, gamma, 'binSVM');
hBTr = mean(hogberTr);
hBTe = mean(hogberTe);
hBTr2 = mean(hogberTr2);
hBte2 = mean(hogberTe2);

[cmberTr, cmbernTe, cmberTr2, cmberTe2] = crossValidation(XTrcnn, yTr, 5, C, gamma, 'multiSVM');
cMTr = mean(cmberTr);
cMTe = mean(cmbernTe);
cMTr2 = mean(cmberTr2);
cMte2 = mean(cmberTe2);

[hmberTr, hmberTe, hmberTr2, hmberTe2] = crossValidation(XTrcnn, yTr, 5, C, gamma, 'multiSVM');
hMTr = mean(hmberTr);
hMBTe = mean(hmberTe);
hMTr2 = mean(hmberTr2);
hMte2 = mean(hmberTe2);
save('data/wsTestSVM.mat');
%%

[aberTr, aberTe, aberTr2, aberTe2] = crossValidation(XTrall, yTr, 5, C, gamma, 'binSVM');
aBTr = mean(aberTr);
aBTe = mean(aberTe);
aBTr2 = mean(aberTr2);
aBte2 = mean(aberTe2);

[amerTr, amerTe, amerTr2, amerTe2] = crossValidation(XTrall, yTr, 5, C, gamma, 'multiSVM');
aMTr = mean(amerTr);
aMTe = mean(amerTe);
aMTr2 = mean(amerTr2);
aMte2 = mean(amerTe2);

save('data/wsTestSVM.mat');
