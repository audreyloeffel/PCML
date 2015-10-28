function [ cost ] = computeCostLogistic( y, tX, beta )
%COMPUTECOST Summary of this function goes here
%   Detailed explanation goes here

%syms n
%sumTemp = symsum(y(n)*tX(n, :)*beta, n, 1, length(y));

s = 0;
for n=1:length(y)
  s = s+ y(n)*tX(n,:)*beta;

end
l = log(1+exp(tX(n,:)*beta));

cost = -(s - l);
end

