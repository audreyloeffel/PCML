function beta = penLogisticRegression(y,tX,alpha,lambda)
%PENLOGISTICREGRESSION
% Written by Meryem M'hamdi and Audrey Loeffel
% - Compute the beta parameter for the penalized logistic regression.
%
% Function: beta = logisticRegression(y, tx, alpha)
% Inputs: y : target data
%         tx: matrix built with the input data
%         alpha: step-size
%         lambda: regularization factor
% Output: beta: parameter optimizing the cost

maxIters = 10000;
N = length(y);
M = size(tX);
beta = zeros(M(2),1);
for k = 1:maxIters
    % Compute the gradient g
    g = tX' * (sigmoid(tX*beta)-y)+ lambda.* beta;
    S = eye(N,N);
    for j = 1: N
        S (j,j) = sigmoid(tX(j,:)*beta) * (1-sigmoid(tX(j,:)*beta));
    end
    % Compute the Hessian H
    H = tX' * S * tX + lambda.* eye(size(beta,1),size(beta,1));
    beta = beta - alpha.*inv(H)* g;
    if g'*g < 1e-5; break; end;
end
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

