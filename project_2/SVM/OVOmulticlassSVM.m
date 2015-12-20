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
    
    function [ yTe_hat ] = multiSVM_onevsone(XTr, yTr, XTe, C, ker)
pair = nchoosek(1:4,2);

yTe_hat = zeros(length(XTe),4);

for i = 1:length(pair)
    lab1 = double(pair(i,1));
    lab2 = double(pair(i,2)); 
    
    idx = find((yTr == lab1) + (yTr == lab2));
    
    tX = XTr(idx,:);
    ty = yTr(idx,:);
    ty(ty == lab1) = 1;
    ty(ty == lab2) = -1;
    
    [~,yTe_pred] = SVM(tX, ty, XTe, C, ker);

    yTe_hat(:,lab1) = yTe_hat(:,lab1) + 1*(yTe_pred == 1)';
    yTe_hat(:,lab2) = yTe_hat(:,lab2) + 1*(yTe_pred == -1)';
end

[~,yTe_hat] = max(yTe_hat,[],2);

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

