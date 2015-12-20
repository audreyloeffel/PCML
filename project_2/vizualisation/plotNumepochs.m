

load('data/numepochs.mat');


%batchsize

data = resultNP

x = data(:, 1);
erryTr = data(:, 4);
erryTe = data(:, 5);
plot(x, erryTr)
hold on;
plot(x, erryTe)
legend('Test error', 'Training error');
xlabel('numepochs');
ylabel('error');