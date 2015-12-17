% [U,S,V] = svd(train.X_hog);
% %Create a new matrix S2 that is a copy of matrix S
% S2 = zeros(size(S,1),size(S,2));
% for i = 1:size(S,1)
%     for j = 1:size(S,2)
%         S2(i,j) = S(i,j);
%     end
% end
% %Set all values in diagonal matrix S less than a certain threshold 0.1 to 0
% %and store the new matrix S in S2
% for i = 1:size(S2,2)
%     if(S2(i,i)<0.1) 
%         S2(i,i)=0;
%     end
% end
% %Get the new X_hog using the new S
% X_hog2 = U*S2*V';

%EigenVectors and Values extraction from correlation matrix
A = train.X_hog;
R = A*A';
[eigVec, eigVal] = eig(R);

eigVal2 = eigVal;
for i = 1:size(eigVal,1)
    if (eigVal(i,i)<0.1)
       eigVal2(i,:) = []; 
    end     
end


X_hog2 =  A' * eigVal;