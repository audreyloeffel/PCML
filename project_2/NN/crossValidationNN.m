function [berTr, berTe, berTr2,berTe2] = crossValidationNN(X, Y, K, neuralFt, numpoch, batchsize, rate)

% split data in K fold
setSeed(1);

N = size(Y,1);
idx = randperm(N);
Nk = floor(N/K);
Y = double(Y);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for k = 1:K
    
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    yTe = Y(idxTe);
    XTe = X(idxTe,:);
    yTr = Y(idxTr);
    XTr = X(idxTr,:);
    
    fprintf('[Cross validation] %i folg\n', k);
        
    [y_hat, ~] = simpleNeuralNetworkOpti(XTr, yTr, XTr, neuralFt, numpoch, batchsize, rate);
    errorTr(k) = cBER(yTr, y_hat);
    errorTr2(k) = ber(yTr, y_hat);
    clear y_hat; clear p_hat;
    [y_hat, ~] = simpleNeuralNetworkOpti(XTr, yTr, XTe, neuralFt, numpoch, batchsize, rate);
    errorTe(k) = cBER(yTe, y_hat);
    errorTe2(k) = ber(yTe, y_hat);
    
    
    
end

berTr = errorTr;
berTe = errorTe;
berTr2 = errorTr2;
berTe2 = errorTe2;

% berTr = mean(errorTr);
% berTe = mean(errorTe);
% berTr2 = mean(errorTr2);
% berTe2 = mean(errorTe2);

end