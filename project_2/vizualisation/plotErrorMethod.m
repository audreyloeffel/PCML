%load('data/wsTestSVM.mat')
load('data/BinarySVM_hog_cnn.mat');

%% SVM Binary
set(gca, 'LooseInset', get(gca, 'TightInset'));
figure;

boxplot([hogberTe' cnnberTe', aberTe(1, 1:4)'],'labels',{'Hog', 'CNN', 'Hog + CNN'});
hx = ylabel('Test error');
hy = xlabel('Features');
set(gca,'fontsize',20,'fontname','Helvetica','xcolor',[0.5 0.5 0.5],'ycolor',[0.5 0.5 0.5]);
set([hx hy], 'fontsize', 20,'color',[0.2 0.2 0.2] );
set(gca, 'ygrid', 'on','GridlineStyle','--');
print -dpdf bin_svm_hog_cnn.pdf

%% SVM Multiclass
set(gca, 'LooseInset', get(gca, 'TightInset'));
figure;
boxplot([hmberTr2' cnnberTr2' amerTr2'],'labels',{'Hog', 'CNN', 'Both'});
hx = ylabel('Test error Multiclass');
hy = xlabel('Model');
set(gca,'fontsize',20,'fontname','Helvetica','xcolor',[0.5 0.5 0.5],'ycolor',[0.5 0.5 0.5]);
set([hx hy], 'fontsize', 20,'color',[0.2 0.2 0.2] );
set(gca, 'ygrid', 'on','GridlineStyle','--');