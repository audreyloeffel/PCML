clear all;
load('resultAlpha');
alpha= aTrTe(:,1);
errTr = aTrTe(:,2);
errTe = aTrTe(:,3);
figure;
scatter(alpha, errTe));
hold on;
plot(alpha, errTr);
