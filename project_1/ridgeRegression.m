function [ beta ] = ridgeRegression( y, tX, lambda )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
lm = lambda.*eye(size(tX, 2));
lm(1,1) = 0;
beta = (tX'*tX + lm)\(tX'*y);

end

