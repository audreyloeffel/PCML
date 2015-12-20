function [ p ] = ber( y, y_hat )
%BER Summary of this function goes here
%   Detailed explanation goes here
c = confusionmat(double(y), double(y_hat));
num_class = size(unique(y), 1);
p = 0;
for i=1:num_class
    p = p + sum(c(i, setdiff(1:num_class, i)))/sum(c(i, :));
end
p = p/num_class;

end
