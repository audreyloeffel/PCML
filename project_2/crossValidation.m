function [berTr, berTe] = crossValidation(X, Y, K, C, gamma, model)

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
     fprintf('avant switch : size x: %f, size y: %f\n', size(XTr,1), size(yTr, 1));
    disp(length(idxTr));        
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
            clear y_hat; clear p_hat;
            [y_hat, p_hat] = SVM(XTr, yTrBin, XTe, C, gamma);
            errorTe(k) = ber(yTeBin, y_hat);
            
        case 'multiSVM'
            fprintf('[Cross validation] %f folg\n', k);
            [y_hat, p_hat] = multiclassSVM(XTr, yTr, gamma, C);
            errorTr(k) = ber(yTr, y_hat);
            clear y_hat; clear p_hat;
            [y_hat, p_hat] = multiclassSVM(XTe, yTe, gamma, C);
            errorTe(k) = ber(yTe, y_hat);
            
        case 'nn'
            fprintf('[Cross validation] %f folg\n', k);
            neuralFt = 10;
            rate = 2;
            fprintf('case: size x: %f, size y: %f\n', length(XTr), length(yTr));
            
            [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTr, neuralFt, rate)
            clear y_hat; clear p_hat;
            [y_hat, p_hat] = simpleNeuralNetwork(XTe, yTe, neuralFt, rate)
            errorTe(k) = ber(yTe, y_hat);
            
        otherwise
            fprintf('It is not a existing model!');
    end

end

berTr = mean(errorTr);
berTe = mean(errorTe);

end