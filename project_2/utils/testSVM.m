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

[cnnberTr, cnnberTe, cnnberTr2, cnnberTe2] = crossValidation(XTrcnn, yTr, 5, C, gamma, 'binSVM');
fprintf('[BINARY CNN SVM] training error: %f\n', cnnberTr);
fprintf('[BINARY CNN SVM] testing error: %f\n', cnnberTe);

[hogberTr, hogberTe, hogberTr2, hogberTe2] = crossValidation(XTrhog, yTr, 5, C, gamma, 'binSVM');
fprintf('[BINARY HOG SVM] training error: %f\n', hogberTr);
fprintf('[BINARY HOG SVM] testing error: %f\n', hogberTe);

[cmberTr, cmberTe, cmberTr2, cmberTe2] = crossValidation(XTrcnn, yTr, 5, C, gamma, 'multiSVM');
fprintf('[MULTI CNN SVM] training error: %f\n', cmberTr);
fprintf('[MULTI CNN SVM] testing error: %f\n', cmberTe);

[hmberTr, hmberTe, hmberTr2, hmberTe2] = crossValidation(XTrcnn, yTr, 5, C, gamma, 'multiSVM');
fprintf('[MULTI HOG SVM] training error: %f\n', hmberTr);
fprintf('[MULTI HOG SVM] testing error: %f\n', hmberTe);
%%
[aberTr, aberTe, aberTr2, aberTe2] = crossValidation(XTr, yTr, 5, C, gamma, 'multiSVM');
fprintf('[MULTI ALL SVM] training error: %f\n', aberTr);
fprintf('[MULTI ALL SVM] testing error: %f\n', aberTe);
