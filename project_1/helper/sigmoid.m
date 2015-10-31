function [ S ] = sigmoid( X )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here
  for i = 1:size(X)
      S(i) = exp(X(i)) / (1 + exp(X(i)));
  end
  S = S';
end

