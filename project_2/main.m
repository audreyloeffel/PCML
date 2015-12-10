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
      
yTr = train.y;
XTr = normalize(XTr);

%% binary {cars, horses, airplane} (1,2,3) -> positive / others (4) -> negative

%transform to binary output
yTr(yTr==4) = 2;
yTr(yTr~=4) = 1;
gamma = 1;
C = 1;
[y_hat, p_hat] = SVM(XTr, yTr, XTe, C, gamma);


%% multiclass {cars: 1, horses: 2, airplane: 3, others: 4}