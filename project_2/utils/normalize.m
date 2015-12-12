function [ X_norm ] = normalize( XTr )
%NORMALIZE ..
%   Detailed explanation goes here
N = size(X,1);
meanX = mean(X);
stdX = std(X);
X_norm = (X - ones(N,1)*meanX)./(ones(N,1)*stdX);

end

