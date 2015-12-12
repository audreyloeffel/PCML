function [ beta ] = leastSquaresGD( y, tX, alpha )
%LEASTSQUARESGD
% Written by Meryem M'hamdi and Audrey Loeffel
% - Compute the beta parameter for the leastsquares regression with
% gradient descent.
%
% Function beta = LeastSquaresGD(y, tx, alpha)
% Inputs: y : target data
%         tx: matrix built with the input data
%         alpha: step-size
% Output: beta: parameter optimizing the cost

    maxIters = 1000;
    D = size(tX(1,:));
    beta = zeros(D(2),1);
    for k = 1:maxIters
        g = computeGradient(y, tX, beta);
        beta = beta - alpha.* g;
        if g'*g < 1e-5;
           break;
        end
    end
end

function [ g ] = computeGradient( y, tX, beta )
% Compute the grad
    e = y - tX*beta;
    g = tX'*e / (-length(y));
end



