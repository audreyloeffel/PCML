%From: http://www.modelselect.inf.ethz.ch/evaluation.php
function error = class_balanced_error_rate (y_train,y_pred)
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
    class1_pred = 0;
    class2_pred = 0;
    class3_pred = 0;
    class4_pred = 0;
    for j = 1:N
        if(y_pred(i) == 1)
            class1_pred = class1_pred + 1;
        elseif(y_pred(i) == 2)
            class2_pred = class2_pred + 1;
        elseif(y_pred(i) == 3)
            class3_pred = class3_pred + 1;
        elseif(y_pred(i) == 4)
            class4_pred = class4_pred + 1;
        end
    end
    error = (1/2) * ((class1_pred/class1_train) + (class2_pred/class2_train)+(class3_pred/class3_train)+(class4_pred/class4_train));
end