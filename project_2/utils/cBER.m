%From: http://www.modelselect.inf.ethz.ch/evaluation.php
function error = cBER (y_train,y_pred)
    N = length(y_train); 
    class1_train = 0;
    class2_train = 0;
    class3_train = 0;
    class4_train = 0;
    for i = 1: N
        if(y_train(i) == 1) 
            class1_train = class1_train + 1;
        elseif (y_train(i)==2)
            class2_train = class2_train + 1;
        elseif (y_train(i)==3)
            class3_train = class3_train + 1;
        elseif(y_train(i)==4)
             class4_train = class4_train + 1;
        end
    end
    class1_error = 0;
    class2_error = 0;
    class3_error = 0;
    class4_error = 0;
    for j = 1:N
        if(y_pred(j) == 1 && y_train(j)~=1)
            class1_error = class1_error + 1;
        elseif(y_pred(j) == 2 && y_train(j)~=2)
            class2_error = class2_error + 1;
        elseif(y_pred(j) == 3 && y_train(j)~=3)
            class3_error = class3_error + 1;
        elseif(y_pred(j) == 4 && y_train(j)~=4)
            class4_error = class4_error + 1;
        end
    end
    error = (1/4) * ((class1_error/class1_train) + (class2_error/class2_train)+(class3_error/class3_train)+(class4_error/class4_train));
end