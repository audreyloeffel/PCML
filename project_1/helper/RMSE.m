function rmse = RMSE(y, pred)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
e = y - pred;
mse = e'*e /(2*length(y));
rmse = sqrt(2*mse);

end
