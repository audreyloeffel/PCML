function [new] = ApplySVD(X)
    [U,S,V] = svd(X);
    N = size(S,2);
    i = 1;
    while(i<=N)
        if(S(i,i)<=2)
            fprintf('\nThe value of i:%d and S(%d,%d)=%d',i,i,i,S(i,i));
            M = i;
            i = N+1;
        end
        i=i+1;
    end
    new = U(:,1:M)*S(1:M,1:M);
end