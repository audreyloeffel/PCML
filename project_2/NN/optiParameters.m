clear all;
load('data/train.mat');
load('data/pca_X_cnn');
yTr = double(train.y);
XTr = x_hat;
K = 4;
numepochs = 30;
batchsize = 50;
rate = 2;
neuralFt = 10;

%% batchsize
fprintf('Batchsize test\n');
rangeBS = linspace(75, 225, 4);
for i = 1:8
    [BatchBerTr(i), BatchBerTe(i), BatchBerTr2(i),BatchBerTe2(i)] = crossValidationNN(XTr, yTr, K, neuralFt, numepochs, rangeBS(i), rate);
    fprintf('[Batchsize = %i] training error: %f test error: %f\n', rangeBS(i), BatchBerTr(i), BatchBerTe(i));
    
end
resultBS = [rangeBS' BatchBerTr' BatchBerTe' BatchBerTr2' BatchBerTe2'];
save('batchsize2.mat', 'resultBS');

%% numepochs
% fprintf('numepochs test\n');
% rangeNP = linspace(10, 80, 8);
% for i = 1:8
%     [NPBerTr(i), NPBerTe(i), NPBerTr2(i),NPBerTe2(i)] = crossValidationNN(XTr, yTr, K, neuralFt, rangeNP(i), batchsize, rate);
%     fprintf('[numepoch = %i] training error: %f test error: %f\n',rangeNP(i), NPBerTr(i), NPBerTe(i));
%     
% end
% resultNP = [rangeNP' NPBerTr' NPBerTe' NPBerTr2' NPBerTe2'];
% save('numepochs.mat', 'resultNP');
% 
% %% rate
% fprintf('rate test\n');
% rangeR = linspace(1, 6, 6);
% for i = 1:8
%     [RBerTr(i), RBerTe(i), RBerTr2(i),RBerTe2(i)] = crossValidationNN(XTr, yTr, K, neuralFt, numepochs, batchsize, rangeR(i));
%     fprintf('[Rate = %i] training error: %f test error: %f\n',rangeR(i), RBerTr(i), RBerTe(i));
%     
% end
% result = [rangeR' RBerTr' RBerTe' RBerTr2' RBerTe2'];
% save('rate.mat', 'result');
% 
% %% neural feature
% % fprintf('neural feature test\n');
% % rangNF = linspace(6, 24, 7);
% % for i = 1:8
% %     [NFBerTr(i), NFBerTe(i), NFBerTr2(i),NFBerTe2(i)] = crossValidationNN(XTr, yTr, K, rangeNF(i), numepochs, batchsize, rate);
% %     fprintf('[Neural features = %i] training error: %f test error: %f\n',RBerTr(i), RBerTe(i));
% %     
% % end
% 
% %%
% save('batchsize.mat', [rangeBS' BatchBerTr' BatcgBerTe' BatchBerTr2' BatcgBerTe2'])