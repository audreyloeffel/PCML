function [ beta ] = logisticRegression( y, tX, alpha )
%LOGISTICREGRESSION 
% Written by Meryem M'hamdi and Audrey Loeffel
% - Compute the beta parameter for the logistic regression.
% 
% Function: beta = logisticRegression(y, tx, alpha)
% Inputs: y : target data
%         tx: matrix built with the input data
%         alpha: step-size
% Output: beta: parameter optimizing the cost

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

function [ g ] = computeGradientLogistic( y, tX, beta )
% Compute the gradient for the case of logistic regression

si = sigmoid(tX*beta);
g = tX'*(si - y);
end

function [ S ] = sigmoid( X )
% Compute the sigma operand

for i = 1:size(X)
    if X(i) > 0
        S(i) = 1 / (1 + exp(-X(i)));
    else
        S(i) = exp(X(i)) / (1 + exp(X(i)));
    end
end
S = S';
end

