function [ S ] = sigmoid( X )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here
   for i = 1:size(X)
         if X(i) > 0
             S(i) = 1 / (1 + exp(-X(i)));
         else
             S(i) = exp(X(i)) / (1 + exp(X(i)));
         end
     end
  S = S';
end

