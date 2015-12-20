function [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTr, XTe, neuralFt, numepochs, batchsize, rate);

num_class = size(unique(yTr),1);
num_feature = size(XTr, 2);

nn = nnsetup([num_feature neuralFt num_class]);
nn.learningRate = rate;
nn.weightPenaltyL2 = 0.0003 ;
nn.activation_function = 'sigm';

% opts.numepochs =  30;   %  Number of full sweeps through data
% opts.batchsize = 500;  %  Take a mean gradient step over this many samples
opts.numepochs = numepochs;
opts.batchsize = batchsize;

idxUsed = opts.batchsize * floor( size(XTr) / opts.batchsize);
XTr= XTr(1:idxUsed,:);
yTr = yTr(1:idxUsed);

for i = 1:num_class
    yLab(:,i) = (yTr == i);
end

[nn, L] = nntrain(nn, XTr, yLab, opts);
y_hat = nnpredict(nn, XTe);
p_hat=0;

end