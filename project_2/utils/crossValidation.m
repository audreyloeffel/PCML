function [berTr, berTe] = crossValidation(X, Y, K, model)

% split data in K fold
setSeed(1);

N = size(Y,1);
idx = randperm(N);
Nk = floor(N/K);
Y = double(Y);
options = statset('UseParallel',1);

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
            yTrBin(yTrBin==4) = 0;
            yTeBin = yTe;
            yTeBin(yTeBin~=4) = 1;
            yTeBin(yTeBin==4) = 0;
            options = statset('UseParallel',1);
            t = templateSVM('BoxConstraint',0.11,'KernelFunction','linear');
            LinAllModel = fitcecoc(XTr,yTrBin,'Learners',t,'Coding','onevsall','Options',options);
            yTr_hat = predict(LinAllModel,XTr);
            yTe_hat = predict(LinAllModel,XTe);
            errorTr(k) = ber(yTrBin, yTr_hat);
            errorTe(k) = ber(yTeBin, yTe_hat);
            
            % model = svmtrain(XTr,yTrBin);
            % y_hat = svmclassify(model,XTr);
            
            % [y_hat, p_hat] = SVM(XTr, yTrBin, XTr, C, gamma);
            %fprintf('size ytr %i, %i, size yhat %i, %i\n',size(yTrBin,1),size(yTrBin,2), size(y_hat,1),size(y_hat,2))
            %model = svmtrain(XTr, yTrBin, 'boxconstraint', 0.1);
            % predTrBin = svmclassify(model, XTr);
            %errorTr(k) = ber(yTrBin, predTrBin);
            %predTeBin = svmclassify(model, XTe);
            %errorTe(k) = ber(yTerBin, predTeBin);
            % clear y_hat; clear p_hat;
            %[y_hat, p_hat] = SVM(XTr, yTrBin, XTe, C, gamma);
            % model = svmtrain(XTr,yTeBin);
            % y_hat = svmclassify(model,XTe);
            %fprintf('size ytr %i, %i, size yhat %i, %i\n',size(yTrBin,1),size(yTrBin,2), size(y_hat,1),size(y_hat,2))
            
            
        case 'multiSVM'
            fprintf('[Cross validation] %i folg\n', k);
            options = statset('UseParallel',1);
            t = templateSVM('BoxConstraint',0.11,'KernelFunction','linear');
            LinAllModel = fitcecoc(XTr,yTr,'Learners',t,'Coding','onevsall','Options',options);
            yTr_hat = predict(LinAllModel,XTr);
            yTe_hat = predict(LinAllModel,XTe);
            errorTr(k) = ber(yTr, yTr_hat);
            errorTe(k) = ber(yTe, yTe_hat);
            
            
%             model = svmtrain(XTr, yTr, 'boxconstraint', 0.1);
%             predTrMul = svmclassify(model, XTr);
%             errorTr(k) = ber(yTr, predTrMul);
%             predTeMul = svmclassify(model, XTe);
%             errorTe(k) = ber(yTe, predTeMul);
%             y_hat = multisvm(XTr, yTr, XTr);
%             
%             errorTr(k) = ber(yTr, y_hat);
%             clear y_hat; clear p_hat;
%             y_hat = multisvm(XTe, yTe, XTe);
%           
%             errorTe(k) = ber(yTe, y_hat);
%             
        case 'bnn'
            fprintf('[Cross validation] %f folg\n', k);
            neuralFt = 20;
            rate = 50;
            yTrBin = yTr;
            yTrBin(yTrBin~=4) = 1;
            yTrBin(yTrBin==4) = 0;
            yTeBin = yTe;
            yTeBin(yTeBin~=4) = 1;
            yTeBin(yTeBin==4) = 0;
            [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTrBin, XTr, neuralFt, rate)
            
            errorTr(k) = ber(yTrBin, y_hat);
            clear y_hat; clear p_hat;
            [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTrBin, XTe, neuralFt, rate)
           
            errorTe(k) = ber(yTeBin, y_hat);
            
        case 'mnn'
            fprintf('[Cross validation] %f folg\n', k);
            neuralFt = 10;
            rate = 2;
          
            [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTr, XTr, neuralFt, rate)
            
            errorTr(k) = ber(yTr, y_hat);
            clear y_hat; clear p_hat;
            [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTr, XTe, neuralFt, rate)
            
            errorTe(k) = ber(yTe, y_hat);
            
        otherwise
            fprintf('It is not a existing model!');
    end
    
end

berTr = errorTr;
berTe = errorTe;

end