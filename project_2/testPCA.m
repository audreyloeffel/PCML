load('data/train.mat');

Xnorm = double(train.X_cnn);
Xnorm(:, end) = [];
k = 500;

[U, mu, vars] = pca(Xnorm);
[yk, x_hat, truc] = pcaApply(Xnorm, U, mu, k);

[berTr, berTe] = crossValidation(x_hat, train.y, 5, 0, 0, 'mnn');
fprintf('[NN multi] training error: %f\n', berTr);
fprintf('[NN multi] testing error: %f\n', berTe);