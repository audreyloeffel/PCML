function [ beta ] = leastSquaresGD( y, tX, alpha )
%LEASTSQUARESGD Summary of this function goes here
%   alpha : step-size

    maxIters = 1000;
    D = size(tX(1,:));
    beta = zeros(D(2),1);
    %beta = gradientDescent(y, tX, alpha, beta, 0, @computeCost, @computeGradient);
    for k = 1:maxIters
        g = computeGradient(y, tX, beta);
        beta = beta - alpha.* g;
        if g'*g < 1e-5;
           break;
        end
    end
end

