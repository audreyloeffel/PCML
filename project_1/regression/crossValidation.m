function [mRMSETr, mRMSETe] = crossValidation(X, Y, K, alpha, lambda, model, degree)

% split data in K fold
setSeed(1);

N = size(Y,1);
idx = randperm(N);
Nk = floor(N/K);
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
    
    if degree == 0    
      
        tXTr = [ones(length(yTr), 1) XTr];    
        tXTe = [ones(length(yTe), 1) XTe]; 
    else
        tXTr = [ones(length(yTr), 1) myPoly(XTr, degree)];    
        tXTe = [ones(length(yTe), 1) myPoly(XTe, degree)]; 
    end
    
    switch model
        case 'rr'            
            beta = ridgeRegression(yTr, tXTr, lambda);
            RMSETr(k) = sqrt(2*MSE(yTr, tXTr, beta));
            RMSETe(k) = sqrt(2*MSE(yTe, tXTe, beta)); 
        case 'lr'
            beta = logisticRegression(yTr, tXTr, alpha);
            RMSETr(k) = sqrt(2*MSE(yTr, tXTr, beta));
            RMSETe(k) = sqrt(2*MSE(yTe, tXTe, beta));
        case 'lq'
            beta = leastSquares(y, tX);
            RMSETr(k) = sqrt(2*MSE(yTr, tXTr, beta));
            RMSETe(k) = sqrt(2*MSE(yTe, tXTe, beta));  
        case 'lqGD'
            beta = leastSquares(y, tX, alpha);
            RMSETr(k) = sqrt(2*MSE(yTr, tXTr, beta));
            RMSETe(k) = sqrt(2*MSE(yTe, tXTe, beta));  
        otherwise
            fprintf('It is not a existing model!');
    end
end

mRMSETr = mean(RMSETr);
mRMSETe = mean(RMSETe);

end