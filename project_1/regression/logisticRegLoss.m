function [L, g, H] = logisticRegLoss(beta, y, tX)

%%Compute cost
L = 0;
S = zeros(size(y));
for i = 1:size(y)
    L = L + y(i)*tX(i,:)*beta - log(1+exp(tX(i,:)*beta));
    S(i,i) = sigmoid(tX(i,:)*beta)*(1-sigmoid(tX(i,:)*beta));
end

L = -L;

%Compute hessian
H = tX'*S*tX;

%Compute gradient
S = sigmoid(tX*beta);
g = tX'*(S - y);

end