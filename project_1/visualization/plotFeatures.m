clear all;
load('Mumbai_regression.mat');

X = normalize(X_train);

%% Plot relevant features
figure;
i = 1;
%subplot(2,1,i);
scatter(X(:, 50), y_train, '.');
xlabel('X\_train');
ylabel('y\_train');
title('Feature 50');
print -dpdf ft50.pdf
% i=i+1;
% subplot(2,1,i);
% scatter(X(:, 21), y_train, '.');
% xlabel('X_train');
% ylabel('y_train');
% title('Feature 21');
% 
% i=i+1;
% subplot(3,3,i);
% scatter(X(:, 38), y_train, '.');
% xlabel('X_train');
% ylabel('y_train');
% title('Feature 38');
% 
% i=i+1;
% subplot(3,3,i);
% scatter(X(:, 46), y_train, '.');
% xlabel('X_train');
% ylabel('y_train');
% title('Feature 46');
% 
% i=i+1;
% subplot(3,3,i);
% scatter(X(:, 50), y_train, '.');
% xlabel('X_train');
% ylabel('y_train');
% title('Feature 50');
% 
% i=i+1;
% subplot(3,3,i);
% scatter(X(:, 56), y_train, '.');
% xlabel('X_train');
% ylabel('y_train');
% title('Feature 56');


% % % Plot all features
% 
% 
% j = 4;
% k = 6;
%     figure;
% for i=1:24
% 
%     subplot(j,k,i);
%     scatter(X(:, i), y_train, '.');
%     xlabel('X_train');
%     ylabel('y_train');
%     title( num2str(i));
% end
%     figure;
% for i=25:49
%     subplot(j,k,(i-24));
%     scatter(X(:, i), y_train, '.');
%     xlabel('X_train');
%     ylabel('y_train');
%     title( num2str(i));
% end
% for i=50:73
%     subplot(j,k,(i-49));
%     scatter(X(:, i), y_train, '.');
%     xlabel('X_train');
%     ylabel('y_train');
%     title( num2str(i));
% end

print -dpdf rlvtFeatures.pdf
