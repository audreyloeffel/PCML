clear all;
load('Mumbai_regression.mat');

% -> see feature 3 and 21

X = normalize(X_train);
figure;
for i = 1:size(X,2) 
   hist(X(:,i), 50);
   title(num2str(i));
   pause;
end