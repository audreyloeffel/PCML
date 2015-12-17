% clear all;
% load train/train.mat;
% 
% %Extract original dataset
% X = train.X_hog;
% X = normalize(X);
% X = ApplyPCA(X); %Apply dimensionality reduction using PCA
% Y = train.y;
% N = length(Y);
% 
% xs0=single(X(1:4000,:)); 
% xs1=single(X(4000:6000,:));
% 
% hs0=single(Y(1:4000)); 
% hs1=single(Y(4000:6000));
% 
% rng(1); % For reproducibility
% tree = fitctree(xs0,hs0,'CrossVal','on');
% 
% numBranches = @(x)sum(x.IsBranch);
% treeNumSplits = cellfun(numBranches, tree.Trained);
% 
% figure;
% histogram(treeNumSplits);
% 
% view(tree.Trained{1},'Mode','graph');

tree1 = prune(tree,'Level',1);
view(tree1,'Mode','Graph');

hsPr0 = predict(tree1,xs0);
fprintf('\nTrain Error calculated using BER:%d',cBER(hs0,hsPr0));
hsPr1 = predict(tree1,xs1);
fprintf('\nTest Error calculated using BER:%d',cBER(hs1,hsPr1));

%Mdl7 = fitctree(X,Y,'MaxNumSplits',7,'CrossVal','on');
%view(Mdl7.Trained{1},'Mode','graph');