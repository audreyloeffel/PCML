clear all;
load('resultAlphaSmall');
selected=(1:8);
alpha= aTrTe(selected,1);
errTr = aTrTe(selected,2);
errTe = aTrTe(selected,3);
figure;
%x(aTrTe(:,1), aTrTe(:,3));
semilogx(alpha, errTr);
hold on;
semilogx(alpha, errTe);
legend('Training error', 'Test error');
xlabel('alpha');
ylabel('RMSE');
grid on;

