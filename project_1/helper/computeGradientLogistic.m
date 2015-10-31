function [ g ] = computeGradientLogistic( y, tX, beta )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    %e = y - tX*beta;
    %g = tX'*e / (-length(y)); %TODO: sigmoid
    
    si = sigmoid(tX*beta);
    g = tX'*(si - y);
end


