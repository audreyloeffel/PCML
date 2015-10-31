function beta = penLogisticRegression(y,tX,alpha,lambda)
maxIters = 10000;
N = length(y);
M = size(tX);
beta = zeros(M(2),1);
  for k = 1:maxIters
      g = tX' * (sigmoid(tX*beta)-y)+ lambda.* beta; 
      S = eye(N,N); 
      for j = 1: N
          S (j,j) = sigmoid(tX(j,:)*beta) * (1-sigmoid(tX(j,:)*beta)); 
      end
      H = tX' * S * tX + lambda.* eye(size(beta,1),size(beta,1));
      beta = beta - alpha.*inv(H)* g;
      if g'*g < 1e-5; break; end;
  end
end