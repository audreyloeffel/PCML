function [berTr, berTe, berTr2, berTe2] = crossValidation(X, Y, K, C, gamma, model)

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
    
    switch model
        case 'binSVM'
            fprintf('[Cross validation] %f folg\n', k);
            yTrBin = yTr;
            yTrBin(yTrBin~=4) = 1;
            yTrBin(yTrBin==4) = -1;
            yTeBin = yTe;
            yTeBin(yTeBin~=4) = 1;
            yTeBin(yTeBin==4) = -1;
            [y_hat, p_hat] = SVM(XTr, yTrBin, XTr, C, gamma);
            errorTr(k) = ber(yTrBin, y_hat);
            errorTe2(k) = ber(yTeBin, y_hat);
            clear y_hat; clear p_hat;
            [y_hat, p_hat] = SVM(XTr, yTrBin, XTe, C, gamma);
             errorTe(k) = mBER(yTeBin, y_hat);
            errorTe2(k) = ber(yTeBin, y_hat);
            
        case 'multiSVM'
            fprintf('[Cross validation] %f folg\n', k);
            [y_hat, p_hat] = multiclassSVM(XTr, yTr, gamma, C);
            errorTr(k) = ber(yTr, y_hat);
            errorTe2(k) = ber(yTr, y_hat);
            clear y_hat; clear p_hat;
            [y_hat, p_hat] = multiclassSVM(XTe, yTe, gamma, C);
            errorTe(k) = mBER(yTe, y_hat);
            errorTe2(k) = ber(yTe, y_hat);
            
        case 'bnn'
            fprintf('[Cross validation] %f folg\n', k);
            neuralFt = 20;
            rate = 50;
            yTrBin = yTr;
            yTrBin(yTrBin~=4) = 1;
            yTrBin(yTrBin==4) = 2;
            yTeBin = yTe;
            yTeBin(yTeBin~=4) = 1;
            yTeBin(yTeBin==4) = 2;
            [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTrBin, XTr, neuralFt, rate)
            errorTr(k) = bBER(yTrBin, y_hat);
            errorTr2(k) = ber(yTrBin, y_hat);
            clear y_hat; clear p_hat;
            [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTrBin, XTe, neuralFt, rate)
            errorTe(k) = bBER(yTeBin, y_hat);
            errorTe2(k) = ber(yTeBin, y_hat);
            
        case 'mnn'
            fprintf('[Cross validation] %f folg\n', k);
            neuralFt = 10;
            rate = 2;
          
            [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTr, XTr, neuralFt, rate)
            errorTr(k) = cBER(yTr, y_hat);
            errorTr2(k) = ber(yTr, y_hat);
            clear y_hat; clear p_hat;
            [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTr, XTe, neuralFt, rate)
            errorTe(k) = cBER(yTe, y_hat);
            errorTe2(k) = ber(yTe, y_hat);
            
        otherwise
            fprintf('It is not a existing model!');
    end
    
end

berTr = errorTr;
berTe = errorTe;
berTr2 =errorTr2;
berTe2 = errorTe2;

end