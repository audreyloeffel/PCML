function [y_hat, p_hat] = OVOmulticlassSVM( XTr, yTr, gamma, C )
%MULTICLASSSVM Summary of this function goes here
%   Detailed explanation goes here
%%
num_class = 4;

for c1 = 1:num_class

    for c2 = (c1+1):num_class
        if c2~=c1 
            fprintf('Classify %i VS %i\n', c1, c2);
            XTrFiltr = XTr(yTr == c1 | yTr ==c2);
            yTrFiltr = yTr(yTr == c1 | yTr ==c2);
            
            for row = 1:size(XTr,1)
                if(XTrFiltr(row,1) == class)
                    XTrFiltr(row,1) = 0;
                else
                    XTrFiltr(row,1) = 1;
                end
        end
    end
    
end
%%
% 
% for class =1:4
%     yTrClass = yTr;
%     for row = 1: length(XTr)
%         if(yTrClass(row,1) == class)
%             yTrClass(row,1) = -1;
%         else
%             yTrClass(row,1) = 1;
%         end
%     end
%         
%     [y_hat(:, class), p_hat(:, class)] = SVM(XTr, yTrClass, XTr, C, gamma);
% end
% 
% 
% % Classify into the clusters
%     for i = 1:length(y_hat)
%         [pred(i,1), idx(i,1)] = max(p_hat(i, :));
%     end
%     
%     y_hat = idx;
%     p_hat = pred;
end

