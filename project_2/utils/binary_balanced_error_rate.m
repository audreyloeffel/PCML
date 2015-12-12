%From: http://www.modelselect.inf.ethz.ch/evaluation.php
function error = binary_balanced_error_rate (y_train,y_pred)
    N = length(y_train); 
    pos_train = 0;
    neg_train = 0;
    for i = 1: N
        if(y_train(i) == 4) 
            neg_train = neg_train + 1;
        end
    end
    pos_train = N - neg_train;
    pos_pred = 0;
    neg_pred = 0;
    for j = 1:N
        if(y_pred(i)==4)
            neg_pred = neg_pred + 1;
        else
            pos_pred = pos_pred + 1;
        end
    end
    error = (1/2) * ((pos_train/pos_pred) + (neg_train/neg_pred));
end