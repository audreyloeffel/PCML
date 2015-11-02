clear all;
load('D:\EPFL\Fall 2015\Pattern Classification and Machine Learning CS-433\project_1\datas\Mumbai_classification.mat');

selected = [1:2];

%Select only some features
X = X_train(:, selected);

% Transform y {-1, 1} to {0, 1}
%Y = (y_train+1)./2;
Y = y_train;
% Transform the datas
proportion = 0.8;
[XTr, yTr, XTe, yTe] = split(Y, X, proportion);

nbTraining = size(XTr, 1);
nbTest = size(XTe, 1);
nbFeature = size(XTr, 2);

XTrnorm = normalize(XTr);
XTenorm = normalize(XTe);

% Logistic Regression Method

tXtr = [ones(nbTraining, 1), XTrnorm];
tXte = [ones(nbTest, 1), XTenorm];

alpha = 0.01;
beta = logisticRegression(yTr, tXtr,alpha);
cost = computeCostLogistic(yTe, tXte, beta);
disp(cost);

% Visualize the effect of classification on Training data using Logistic Regression
    % create a n?D meshgrid of values of heights and weights
    ndim = size(XTrnorm,2);
    I=cell(size(XTrnorm,2),1);
    % construct the neighborhood
    for di=1:ndim
        I{di}= min(XTrnorm(:,di)):.01:max(XTrnorm(:,di));
    end
    [I{1:ndim}]=ndgrid(I{:});
    pred =  zeros(size(I{1},1),size(I{1},2));
    for i = 1:size(I{1},1)
        for j = 1:size(I{1},2)
            tX = I{1}(:);
            for k = 2:ndim
               tX = cat(2,tX,I{k}(:));
            end
            tX = [1,tX(i,j)];
            pred(i,j) = sigmoid(tX * beta);
        end
    end 
    
    pred = reshape(pred,[size(hx,1),size(wx,2)]);
    contourf(hx, wx, pred, 1);
    
    hold on;
    myBlue = [0.06 0.06 1];
    myRed = [1 0.06 0.06];

    maxIters = size(XTrnorm,1);
    j = 0;
    for i = 1:maxIters
       if(yTr(i)==1)
          j=j+1;
       end
    end  
    category1 = zeros(j,1);
    category2 = zeros(maxIters-j,1);
    k_category1 = 0;
    k_category2 = 0;
    for i = 1:maxIters
      if(yTr(i)==1) 
          k_category1 = k_category1 + 1;
          category1(k_category1) = i;
      else 
          k_category2 = k_category2 +1;
          category2(k_category2) = i;
      end
     end  

    plot(XTrnorm(category1,1), XTrnorm(category1,2),'xr','color',myRed,'linewidth', 2, 'markerfacecolor', myRed);
    hold on;
    plot(XTrnorm(category2,1), XTrnorm(category2,2),'or','color', myBlue,'linewidth', 2, 'markerfacecolor', myBlue);
    xlabel('height');
    ylabel('weight');
    xlim([min(h) max(h)]);
    ylim([min(w) max(w)]);
    grid on;
    

    
%{
% Visualize the effect of classification on Testing data using Logistic Regression
    % create a 2?D meshgrid of values of heights and weights
    h = [min(XTenorm(:,1)):.01:max(XTenorm(:,1))];
    w = [min(XTenorm(:,2)):.01:max(XTenorm(:,2))];
    [hx, wx] = meshgrid(h,w);
    
    %Create prediction
    pred =  zeros(size(hx,1),size(wx,2));
    for i = 1:size(hx,1)
        for j = 1:size(wx,2)
            tX = [1,hx(i,j),wx(i,j)];
            pred(i,j) = sigmoid(tX * beta);
        end
    end 
    pred = reshape(pred,[size(hx,1),size(wx,2)]);
    

    
% Penalized Logistic Regression Method
lambda = 4;
betaPen = penLogisticRegression(yTr, tXtr, alpha, lambda);
costPen = computeCostLogistic(yTe, tXte, betaPen);
disp(costPen);



% Visualize the effect of classification on Training data Penalized Logistic Regression
    % create a 2?D meshgrid of values of heights and weights
    h = [min(XTrnorm(:,1)):.01:max(XTrnorm(:,1))];
    w = [min(XTrnorm(:,2)):.01:max(XTrnorm(:,2))];
    [hx, wx] = meshgrid(h,w);
    
    %Create prediction
    pred =  zeros(size(hx,1),size(wx,2));
    for i = 1:size(hx,1)
        for j = 1:size(wx,2)
            tX = [1,hx(i,j),wx(i,j)];
            pred(i,j) = sigmoid(tX * beta);
        end
    end 
    pred = reshape(pred,[size(hx,1),size(wx,2)]);


% Visualize the effect of classification on Testing data Penalized Logistic Regression
    % create a 2?D meshgrid of values of heights and weights
    h = [min(XTenorm(:,1)):.01:max(XTenorm(:,1))];
    w = [min(XTenorm(:,2)):.01:max(XTenorm(:,2))];
    [hx, wx] = meshgrid(h,w);
    
    %Create prediction
    pred =  zeros(size(hx,1),size(wx,2));
    for i = 1:size(hx,1)
        for j = 1:size(wx,2)
            tX = [1,hx(i,j),wx(i,j)];
            pred(i,j) = sigmoid(tX * beta);
        end
    end 
    pred = reshape(pred,[size(hx,1),size(wx,2)]);
%}