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

hmberTr2 = [0.2601921, 0.2498001, 0.2409121, 0.2510002];
cnnberTr2= [0.1409712, 0.1509212, 0.1498124, 0.1612129];
amerTr2 =  [0.1701231, 0.1696481, 0.1682302, 0.1730181];

boxplot([hmberTr2' cnnberTr2' amerTr2'],'labels',{'Hog', 'CNN', 'Both'});
hx = ylabel('Test error Multiclass');
hy = xlabel('Model');
set(gca,'fontsize',20,'fontname','Helvetica','xcolor',[0.5 0.5 0.5],'ycolor',[0.5 0.5 0.5]);
set([hx hy], 'fontsize', 20,'color',[0.2 0.2 0.2] );
set(gca, 'ygrid', 'on','GridlineStyle','--');
print -dpdf multi_svm_hog_cnn.pdf

%% NN  Binary
set(gca, 'LooseInset', get(gca, 'TightInset'));
figure;
hmberTr2 = [0.3101321, 0.3091023, 0.2909327, 0.2813762];
cnnberTr2= [0.1609712, 0.1609212, 0.178124, 0.1822129];
amerTr2 =  [0.2401231, 0.2261421, 0.2387012, 0.232655]; 
boxplot([hmberTr2' cnnberTr2', amerTr2'],'labels',{'Hog', 'CNN', 'Hog + CNN'});
hx = ylabel('Test error');
hy = xlabel('Features');
set(gca,'fontsize',20,'fontname','Helvetica','xcolor',[0.5 0.5 0.5],'ycolor',[0.5 0.5 0.5]);
set([hx hy], 'fontsize', 20,'color',[0.2 0.2 0.2] );
set(gca, 'ygrid', 'on','GridlineStyle','--');
print -dpdf bin_nn_hog_cnn.pdf

%% NN Multiclass
set(gca, 'LooseInset', get(gca, 'TightInset'));
figure;

hmberTr2 = [0.2801321, 0.2891023, 0.270027, 0.27002];
cnnberTr2= [0.1509712, 0.1709212, 0.148124, 0.1522129];
amerTr2 =  [0.2001231, 0.2261421, 0.2187012, 0.213655]; 

boxplot([hmberTr2' cnnberTr2' amerTr2'],'labels',{'Hog', 'CNN', 'Both'});
hx = ylabel('Test error Multiclass');
hy = xlabel('Model');
set(gca,'fontsize',20,'fontname','Helvetica','xcolor',[0.5 0.5 0.5],'ycolor',[0.5 0.5 0.5]);
set([hx hy], 'fontsize', 20,'color',[0.2 0.2 0.2] );
set(gca, 'ygrid', 'on','GridlineStyle','--');
print -dpdf multi_nn_hog_cnn.pdf

