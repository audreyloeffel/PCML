%load('data/wsTestSVM.mat')

%% SVM Binary
set(gca, 'LooseInset', get(gca, 'TightInset'));
figure;

boxplot([hogberTe2' cnnberTe2' aberTe2'],'labels',{'Hog', 'CNN', 'Both'});
hx = ylabel('Test error binary');
hy = xlabel('Features');
set(gca,'fontsize',20,'fontname','Helvetica','xcolor',[0.5 0.5 0.5],'ycolor',[0.5 0.5 0.5]);
set([hx hy], 'fontsize', 20,'color',[0.2 0.2 0.2] );
set(gca, 'ygrid', 'on','GridlineStyle','--');

%% SVM Multiclass
set(gca, 'LooseInset', get(gca, 'TightInset'));
figure;
boxplot([hmberTr2' cnnberTr2' amerTr2'],'labels',{'Hog', 'CNN', 'Both'});
hx = ylabel('Test error Multiclass');
hy = xlabel('Model');
set(gca,'fontsize',20,'fontname','Helvetica','xcolor',[0.5 0.5 0.5],'ycolor',[0.5 0.5 0.5]);
set([hx hy], 'fontsize', 20,'color',[0.2 0.2 0.2] );
set(gca, 'ygrid', 'on','GridlineStyle','--');