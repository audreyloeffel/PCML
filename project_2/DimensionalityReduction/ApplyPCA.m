function [train_out] = ApplyPCA(X)
    train_out = X'; % save original data
    mn = mean(train_out);
    train_out = bsxfun(@minus,train_out,mn); % substract mean
    [coefs,scores,variances] = pca(train_out); % PCA
    pervar = cumsum(variances) / sum(variances);
    dims = max(find(pervar < 0.99));
    train_out = (train_out)'*coefs(:,1:dims); % Keep these many dimensions
end