function [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTr, XTe, neuralFt, rate)


num_class = size(unique(yTr),1);
fprintf('number of class: %f\n', num_class);
num_feature = size(XTr, 2);

nn = nnsetup([num_feature neuralFt num_class]);
nn.learningRate = rate;

opts.numepochs =  30;   %  Number of full sweeps through data
opts.batchsize = 500;  %  Take a mean gradient step over this many samples

idxUsed = opts.batchsize * floor( size(XTr) / opts.batchsize);
XTr= XTr(1:idxUsed,:);
yTr = yTr(1:idxUsed);

for i = 1:num_class
    yLab(:,i) = (yTr == i);
end


[nn, L] = nntrain(nn, XTr, yLab, opts);

% nn.testing = 1;
% nn = nnff(nn, XTe, zeros(size(XTe,1), nn.size(end)));
% nn.testing = 0;
% 
% % predict on the test set
% nnPred = nn.a{end};

% get the most likely class
%[~,y_hat] = max(nnPred,[],2);


% y_hat2 = nn.a{end};
 y_hat = nnpredict(nn, XTe);
% 
 p_hat=0;



end