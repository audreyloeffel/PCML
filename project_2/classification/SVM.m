function [y_hat, p_hat] = SVM(XTr, yTr, XTe, C, gamma)


% Choice of kernel 
%K = rbf_kernel(XTr, XTr, gamma);
K = linear_kernel(XTr, XTr);

% SMO
[alphas, beta0] = SMO(K, yTr, 0.0002);

% compute predictions
tX_pred = XTe;
SV_inds = find(alphas>0);
X_SV = XTr(SV_inds, :);
y_SV = yTr(SV_inds, :);
alphas_SV = alphas(SV_inds);
kernel_pred = linear_kernel(tX_pred, X_SV);
p_hat = kernel_pred * (alphas_SV .* y_SV) + beta0;

 % Predications are {-1,1} but we want {1,2}
y_hat = zeros(size(p_hat));
y_hat(p_hat<0) = 2;
y_hat(p_hat>=0) = 1;

end