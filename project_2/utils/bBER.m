%From: http:b//www.modelselect.inf.ethz.ch/evaluation.php
function error  = bBER (y_train,y_pred)
    N = length(y_train);
    neg_train = 0;
    for i = 1: N
        if(y_train(i) == 0) 
            neg_train = neg_train + 1;
        end
    end
    pos_train = N - neg_train;
    pos_error = 0;
    neg_error = 0;
    for j = 1:N
        if(y_pred(j)==1 && y_train(j)==0)
            neg_error = neg_error + 1;
            fprintf('\nj=%d and y_pred(j)=%d and y_train(j)=%d',j, y_pred(j),y_train(j));
        elseif(y_pred(j)==0 && y_train(j)==1)
            pos_error = pos_error + 1;
        end
    end
    error = (1/2) * ((neg_error/neg_train) + (pos_error/pos_train));
end