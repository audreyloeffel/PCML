load('data/train.m');
%%
y = double(train.y);
hist(y, unique(y));
xlabel('Class');
ylabel('Frequency');

set(gca, 'ygrid', 'on','GridlineStyle','--');

print -dpdf hist_y.pdf