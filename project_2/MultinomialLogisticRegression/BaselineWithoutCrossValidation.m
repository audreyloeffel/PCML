% Written by Audrey Loeffel and Meryem M'hamdi, EPFL for PCML Fall 2015
% all rights reserved

% This code applies LogisticRegression to our multi-class classification problem,
% predicts test errors using cross validation method and predicts output 
% for test data using this model 

clear all;
load train/train.mat;

%Extract original dataset
X = normalize(train.X_hog);
Y_original = train.y;
N = length(Y_original);
Y = zeros(N,1);

alpha = 0.000005;
nb_classes = 4;

probaY1r = zeros(N,nb_classes);
%Loop over the possible partitions of multi-class classification logistic regression
for q = 1:nb_classes
    %Assign positive and negative classes to reduce our multi-class problem to
    %binary classification problem
    for p = 1:N
        % Our Positive class 
        if (Y_original(p)==q)
            Y(p)=1;
        % Our Negative class
        else Y(p)=0;
        end
    end

    tX = [ones(N,1) X];

    %% Applying the method Logistic Regression 
    beta = logisticRegression(Y, tX,alpha); 

    %%Prediction for train data
    predr = tX*beta;
    probaY1r(:,q) = sigmoid(predr); % return the probability for the point
    yClassr = zeros(N,1);
end
for i = 1:N
    [maxprobab, index]= max(probaY1r(i,:));
     yClassr(i,1) = index;
end

%Calculate the error
lossr = cBER(double(Y_original),yClassr);
fprintf('\nCost using BER is %d',lossr);

 



