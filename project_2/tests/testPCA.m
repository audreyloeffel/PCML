%
% Apply PCA on Hog and CNN features and save them in a new file.
%

load('data/train.mat');
fprintf('PCA on CNN features ...');
Xnorm = double(train.X_cnn);
Xnorm(:, end) = [];
k = 500;

[U, mu, vars] = pca(Xnorm);
[yk, Xcnn, truc] = pcaApply(Xnorm, U, mu, k);
save('data/pca_Xcnn.mat', 'Xcnn');

fprintf('PCA on Hog features ...');
Xnorm = double(train.X_hog);
Xnorm(:, end) = [];
k = 400;

[U, mu, vars] = pca(Xnorm);
[yk, Xhog, truc] = pcaApply(Xnorm, U, mu, k);
save('data/pca_Xhog.mat', 'Xhog');

