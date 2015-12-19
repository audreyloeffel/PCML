function [train_out] = ApplyPCA(X)
%     disp(size(X));
%     train_out = X';
%     mn = mean(train_out);
%     train_out = bsxfun(@minus,train_out,mn); % substract mean
%      [coefs,scores,variances] = pca(train_out); % PCA
%     pervar = cumsum(variances) / sum(variances);
%     dims = max(find(pervar < 0.99));
%     train_out = (train_out)'*coefs(:,1:dims); % Keep these many dimensions
%     disp(size(scores));
%     disp(size(coefs'));
%    train_out = scores * coefs';

k = 5000; % final number of features

[U,mu,vars] = pca(X');
[yk, x_hat, sq] = pcaApply( X, U, mu, k );
train_out = x_hat
end