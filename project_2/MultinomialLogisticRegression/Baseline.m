%Applying Stochastic Gradient Descent Approach to our hog data
  clear all

  % run grid search first to get the baseline
  %gridSearch;

  % Load data and comvert it to the metrics system
load train/train.mat;
  
  % normalize features (store the mean and variance)
  x = train.X_hog;
  x = normalize(x);
  % Form (y,tX) to get regression data in matrix form
  y = train.y;
  N = length(y);
  tX = [ones(N,1) x];


  % algorithm parametes
  maxIters = 1000;
  alpha = 0.1;
  converged = 0;

  % initialize
  beta = [0; 0];

  % iterate
  fprintf('Starting iterations, press Ctrl+c to break\n');
  fprintf('L  beta0 beta1\n');
  N = length(y);
  count = 0;
  for k = 1:maxIters
    % INSERT YOUR FUNCTION FOR COMPUTING GRADIENT
    %g = 0;
    g = computeGradientLogistic(y,tX,beta);
    beta = beta - alpha * g;
    if g'*g < 1e-5; break; end;
    % INSERT YOUR FUNCTION FOR COMPUTING COST FUNCTION
    L = 0;

    % INSERT GRADIENT DESCENT UPDATE TO FIND BETA
   % beta = beta;

    % INSERT CODE FOR CONVERGENCE

    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;

    % print
    fprintf('%.2f  %.2f %.2f\n', L, beta(1), beta(2));
    count = count +1;
    % Overlay on the contour plot
    % For this to work you first have to run grid Search
    subplot(121);
    plot(beta(1), beta(2), 'o', 'color', 0.7*[1 1 1], 'markersize', 12);
    pause(.5) % wait half a second

    % visualize function f on the data
    subplot(122);
    x = [1.2:.01:2]; % height from 1m to 2m
    x_normalized = (x - meanX)./stdX;
    f = beta(1) + beta(2).*x_normalized;
    plot(height, weight,'.');
    hold on;
    plot(x,f,'r-');
    hx = xlabel('x');
    hy = ylabel('y');
    hold off;
  end
  
  fprintf('count =%d\n',count);

