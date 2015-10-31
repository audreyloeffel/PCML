function normX = normalize( X )
%NORMALIZE Normalize the values X
%   Detailed explanation goes here

meanX = mean(X);
stdX = std(X);
temp = zeros(size(X));
for i = 1:size(X,2)
    temp(:, i) = (X(:, i) - meanX(i))./stdX(i); 
end
normX = temp;
end

