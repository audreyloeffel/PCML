function [ beta ] = ridgeRegression( y, tX, lambda )
%RIDGEREGRESSION 
% Written by Meryem M'hamdi and Audrey Loeffel
% - Compute the beta parameter for the ridge .
% 
% Function: beta = logisticRegression(y, tx, alpha)
% Inputs: y : target data
%         tx: matrix built with the input data
%         lambda: regularization factor
% Output: beta: parameter optimizing the cost

lm = lambda.*eye(size(tX, 2));
lm(1,1) = 0;
beta = (tX'*tX + lm)\(tX'*y);

end

