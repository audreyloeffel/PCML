function X_norm = normalizeMe( X )
%NORMALIZE ..
%   Detailed explanation goes here
% N = size(X,1);
% meanX = mean(X);
% stdX = std(X);
% X_norm = (X - ones(N,1)*meanX)*diag(1./stdX);

meanX = mean(X);
stdX = std(X);
temp = zeros(size(X));
for i = 1:size(X,2)
    temp(:, i) = (X(:, i) - meanX(i))./stdX(i); 
end
X_norm = temp;

end

