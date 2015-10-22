function [ beta ] = penLogisticRegression( y, tX, alpha, lambda )
%PENLOGISTICREGRESSION Summary of this function goes here
%   alpha is the step size for gradient descent, lambda is the regularization parameter

%beta0 = zeros(length(y), 1);
%beta = gradientDescent(y, tX, alpha, lambda, beta0, @computeCostLogistic, @computeGradientLogistic);

maxIters = 1000;
D = size(tX(1,:));
beta = zeros(D(2),1);
for k = 1:maxIters
    [L, g ,H] = logisticRegLoss(beta, y, tX);
    pen = H\g;
    beta = beta - alpha.* pen;
    if g'*g < 1e-5;
        break;
    end
end

end

