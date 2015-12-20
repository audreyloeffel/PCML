load('data/batchsize.mat');

data = [resultBS']
load('data/batchsize2.mat');
data = [data resultBS'];
data = data';
data = sortrows(data, 1);
x = data(:, 1);
erryTr = data(:, 4);
erryTe = data(:, 5);
plot(x, erryTr)
hold on;
plot(x, erryTe)
legend('Test error', 'Training error');
xlabel('batchsize');
ylabel('error');