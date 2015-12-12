function beta = leastSquares( y, tX )
%LEASTSQUARES
% Written by Meryem M'hamdi and Audrey Loeffel
% - Compute the beta parameter for the leastsquares regression
%
% Function beta = LeastSquares(y, tx)
% Inputs: y : target data
%         tx: matrix built with the input data 
% Output: beta: parameter optimizing the cost

beta = (tX' * tX) \ (tX' *y);
end

