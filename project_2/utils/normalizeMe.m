function X_norm = normalizeMe( X )
%NORMALIZE ..
%   Detailed explanation goes here
N = size(X,1);
meanX = mean(X);
stdX = std(X);
X_norm = (X - ones(N,1)*meanX)*diag(1./stdX);

end

