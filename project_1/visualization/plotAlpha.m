clear all;
load('resultAlpha');
selected=(1:20);
alpha= aTrTe(selected,1);
errTr = aTrTe(selected,2);
errTe = aTrTe(selected,3);
figure;
%x(aTrTe(:,1), aTrTe(:,3));
plot(alpha, errTr, alpha, errTe);

