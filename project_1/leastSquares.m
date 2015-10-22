function beta = leastSquares( y, tX )
%LEASTMEAN Summary of this function goes here
%   Detailed explanation goes here

beta = (tX' * tX) \ (tX' *y);
end

