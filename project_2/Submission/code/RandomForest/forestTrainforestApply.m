clear all;
load train/train.mat;

%Extract original dataset
X = train.X_hog;
X = normalize(X);
X = ApplyPCA(X); %Apply dimensionality reduction using PCA
Y = train.y;
N = length(Y);

xs0=single(X(1:4000,:)); 
xs1=single(X(4000:6000,:));

%<<<<<<<<<<<<<<Multi-Class Classification>>>>>>>>>>>>>>>>>

hs0=single(Y(1:4000)); 
hs1=single(Y(4000:6000));

%Tree Parameters: Maximum Depth, Number of Trees, number of features to
%sample for each node split
pTrain={'maxDepth',50,'F1',1245,'M',150,'minChild',4};
forest=forestTrain(xs0,hs0,pTrain{:}); 

%Apply Random Forest Model to train data
hsPr0 = forestApply(xs0,forest);
fprintf('\nTrain Error calculated using BER:%d',cBER(hs0,hsPr0));

%Apply Random Forest Model to test data
hsPr1 = forestApply(xs1,forest);
fprintf('\nTest Error calculated using BER:%d',cBER(hs1,hsPr1));

%Estimating Error using Mean
e0=mean(hsPr0~=hs0); e1=mean(hsPr1~=hs1);
fprintf('\nerrors trn=%f tst=%f\n',e0,e1); figure(1);

%Visualizing the train and test data after classification
subplot(2,2,1); visualizeData(xs0,2,hs0);
subplot(2,2,2); visualizeData(xs0,2,hsPr0);
subplot(2,2,3); visualizeData(xs1,2,hs1);
subplot(2,2,4); visualizeData(xs1,2,hsPr1);


%<<<<<<<<<<<<<<Binary Classification>>>>>>>>>>>>>>>>>>>>>>
Y_bin = zeros(N,1);
%Assign positive classes to 1,2,3 and negative classe to 4 
for p = 1:N
    % Our Negative class 
    if (Y(p)==4)
        Y_bin(p)=1;
    % Our Negative class
    else Y_bin(p)=2;
    end
end

hs_bin0=single(Y_bin(1:4000)); 
hs_bin1=single(Y_bin(4000:6000));

%Tree Parameters: Maximum Depth, Number of Trees, number of features to
%sample for each node split
pTrain={'maxDepth',50,'F1',1245,'M',50,'minChild',1};
forest1=forestTrain(xs0,hs_bin0,pTrain{:});

%Apply Random Forest Model to train data
hsPr_bin0 = forestApply(xs0,forest1);

%Converting hs_bin0 from {1,2} to {0,1} 
for p = 1:4000
    % Our Negative class 
    if (hs_bin0(p)==1)
        hs_bin0(p)= 0;
    % Our Negative class
    else hs_bin0(p)=1;
    end
end

%Converting hs_bin0 from {1,2} to {0,1} 
for p = 1:4000
    % Our Negative class 
    if (hsPr_bin0(p)==1)
        hsPr_bin0(p)= 0;
    % Our Negative class
    else hsPr_bin0(p)=1;
    end
end

fprintf('\nTrain Error calculated using BER:%d',bBER(hs_bin0,hsPr_bin0));

%Apply Random Forest Model to test data
hsPr_bin1 = forestApply(xs1,forest1);

%Converting hs_bin0 from {1,2} to {0,1} 
for p = 1:2001
    % Our Negative class 
    if (hs_bin1(p)==1)
        hs_bin1(p)= 0;
    % Our Negative class
    else hs_bin1(p)=1;
    end
end

%Converting hs_bin0 from {1,2} to {0,1} 
for p = 1:2001
    % Our Negative class 
    if (hsPr_bin1(p)==1)
        hsPr_bin1(p)= 0;
    % Our Negative class
    else hsPr_bin1(p)=1;
    end
end

fprintf('\nTest Error calculated using BER:%d',bBER(hs_bin1,hsPr_bin1));

%Estimating Error using Mean
e0=mean(hsPr_bin0~=hs_bin0); e1=mean(hsPr_bin1~=hs_bin1);
fprintf('\nerrors trn=%f tst=%f\n',e0,e1); figure(1);

%Visualizing the train and test data after classification
subplot(2,2,1); visualizeData(xs0,2,hs_bin0);
subplot(2,2,2); visualizeData(xs0,2,hsPr_bin0);
subplot(2,2,3); visualizeData(xs1,2,hs_bin1);
subplot(2,2,4); visualizeData(xs1,2,hsPr_bin1);

%% visualize samples and their predictions (test set)
figure;
for i=5000:5020 %Trying for only a subset of 20 images  
    clf();

    img = imread( sprintf('train%05d.jpg', i) );
    imshow(img);


    % show if it is classified as pos or neg, and true label
    title(sprintf('True Label: %d, Binary Pred: %d, Class Pred:%d', Y(i),hsPr_bin1(i-4000) ,hsPr1(i-4000)));
    
    %show
    pause;  % wait for keydo that then,Â 
end
