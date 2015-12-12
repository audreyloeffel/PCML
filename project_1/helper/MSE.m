function mse = MSE(y, tX, beta)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
e = y - tX*beta;
mse = e'*e /(2*length(y));

end

