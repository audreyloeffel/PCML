function L = rmse(y, tX, beta)
    e = y - tX*beta;
    L = sqrt(e'*e /(2*length(y)));
end