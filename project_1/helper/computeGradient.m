function [ g ] = computeGradient( y, tX, beta )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
% N=length(y);
% D=size(tX);
% 
% fprintf('MSE polynomial y: %d , tX: %d, beta: %d size of tX*beta: %d ', N, D(2), length(beta), length(tX*beta));
% disp(tX*beta);

    e = y - tX*beta;
    g = tX'*e / (-length(y));
end


