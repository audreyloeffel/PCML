clear all;
load('data/train.mat');

% parameters 1: hog, 2: cnn, 3: both
featuresFrom = 1;

switch featuresFrom
    case 1
        disp('Hog feature')
        XTr = train.X_hog;
    case 2
        disp('CNN feature')
        XTr = train.X_cnn;
    case 3
        disp('Both Hog and CNN')
        XTr = [train.X_hog, train.X_cnn];
    otherwise
        disp('Error')
        XTr = zeros(length(train.X_hog));
end
      
yTr = double(train.y);
XTr = normalizeMe(XTr);

gamma = 1;
C = 0.0002;
[berTr, berTe] = crossValidation(XTr, yTr, 5, C, gamma, 'binSVM');
fprintf('[BINARY SVM] training error: %f\n', berTr);
fprintf('[BINARY SVM] testing error: %f\n', berTe);

[berTr, berTe] = crossValidation(XTr, yTr, 5, C, gamma, 'multiSVM');
fprintf('[MULTI SVM] training error: %f\n', berTr);
fprintf('[MULTI SVM] testing error: %f\n', berTe);

%% binary {cars, horses, airplane} (1,2,3) -> positive / others (4) -> negative

%transform to binary output
yTrBin = yTr;
yTrBin(yTrBin~=4) = 1;
yTrBin(yTrBin==4) = -1;
gamma = 1;
C = 1;
[y_hat, p_hat] = SVM(XTr, yTrBin, XTr, C, gamma);
errorBin = ber(yTrBin, y_hat);

fprintf('[SVM] Binary class error: %f\n', errorBin);

%% multiclass {cars: 1, horses: 2, airplane: 3, others: 4}
gamma = 1;
C = 1;
[y_hat, p_hat] = multiclassSVM(XTr, yTr, gamma, C);
disp(size(y_hat));
disp(y_hat);
errorMulti = ber(yTr, y_hat);
fprintf('[SVM] Multiclass error: %f\n', errorMulti);

