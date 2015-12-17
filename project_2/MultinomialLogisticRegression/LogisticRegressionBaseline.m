% Written by Audrey Loeffel and Meryem M'hamdi, EPFL for PCML Fall 2015
% all rights reserved

% This code applies LogisticRegression to our multi-class classification problem,
% predicts test errors using cross validation method and predicts output 
% for test data using this model 

clear all;
load train/train.mat;

%Extract original dataset
X_original = train.X_hog;
X_original = normalize(X_original);
Y_original = train.y;
X = X_original;
N = length(Y_original);
Y = zeros(N,1);

% split data in K fold 
setSeed(1);
K = 4; 
N = size(Y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

% K-fold cross validation
alpha = [0.000005;0.001;0.01];
lossr = zeros(length(alpha),1);
losse = zeros(length(alpha),1);
nb_classes = 4;
for m = 1:length(alpha)
    fprintf('\nAlpha=%d',alpha(m));
    for k = 1:K
        tic;
        fprintf('\nK=%d',k);
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
            
            % get k'th subgroup in test, others in train
            idxTe = idxCV(k,:);
            idxTr = idxCV([1:k-1 k+1:end],:);
            idxTr = idxTr(:);
            yTe = Y(idxTe);
            XTe = X(idxTe,:);
            yTr = Y(idxTr);
            XTr = X(idxTr,:);
            yTr_original = Y_original(idxTr);
            yTe_original = Y_original(idxTe);

            nbTraining = size(XTr, 1);
            nbTest = size(XTe, 1);
            nbFeature = size(XTr, 2);

            tXtr = [ones(nbTraining, 1), XTr];
            tXte = [ones(nbTest, 1), XTe];

            %% Applying the method Logistic Regression 
            beta = logisticRegression(yTr, tXtr,alpha(m)); 

            %%Prediction for train data
            predr = tXtr*beta;
            probaY1r(:,q) = sigmoid(predr); % return the probability for the point
            yClassr = zeros(nbTraining,1);

            %%Prediction for test data
            prede = tXte*beta;
            probaY1e(:,q) = sigmoid(prede); % return the probability for the point
            yClasse = zeros(nbTest,1);
        end
        for i = 1:nbTraining
            [maxprobab,index]= max(probaY1r(i,:));
             yClassr(i,1) = index;
        end
        for i = 1:nbTest
            [maxprobab,index]= max(probaY1e(i,:));
             yClasse(i,1) = index;
        end
        
        %Calculate the error
        lossr(k) = cBER(double(yTr_original),yClassr);
        fprintf('\nCost using BER is %d',lossr(k));

         %Calculate the error
        losse(k) = cBER(double(yTe_original),yClasse);
        fprintf('\nCost using BER is %d',losse(k));
    TimeSpent = toc;
    fprintf('\nTime spent:%d',TimeSpent);
    end
end

% %plot
% figure;
% plot(alpha, lossr,alpha,losse);
% legend('Train Error','Test Error');
% xlabel('values of alpha');
% ylabel('error');
% title('Logistic Regression using 4-fold Cross Validation');
% grid on;

