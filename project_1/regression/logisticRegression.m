function [ beta ] = logisticRegression( y, tX, alpha )
%LOGISTICREGRESSION Summary of this function goes here
%   Alpha is the step-size, in case of the gradient descent

%beta0 = zeros(length(y), 1);
%beta = gradientDescent(y, tX, alpha, 0, beta0, @computeCostLogistic, @computeGradientLogistic);
 maxIters = 1000;
    D = size(tX(1,:));
    beta = zeros(D(2),1);
    for k = 1:maxIters
        g = computeGradientLogistic(y, tX, beta);
        beta = beta - alpha.* g;
        if g'*g < 1e-5;
           break;
        end
    end
end

