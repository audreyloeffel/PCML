function [ K ] = rbf_kernel( X1, X2,  gamma)
%RBF_KERNEL Compute the radial basis function kernel matrix.

% TODO: implement the RBF kernel
for i=1:size(X1,1)
    for j=1:size(X2,1)
        K(i,j) = exp(-gamma*sum((X1(i,:)-X2(j,:)).^2));
    end
    if mod(i,1000)==0
        fprintf('%d \n',i);
    end
end
end

