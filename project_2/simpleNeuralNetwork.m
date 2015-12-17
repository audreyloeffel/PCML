function [y_hat, p_hat] = simpleNeuralNetwork(XTr, yTr, neuralFt, rate)

num_class = size(unique(yTr),1);
num_feature = size(XTr, 2);

nn = nnsetup([num_feature neuralFt num_class]);


opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
fprintf('nn: size x: %f, size y: %f\n', length(XTr), length(yTr));
[nn, L] = nntrain(nn, XTr, yTr, opts);

%% 


end