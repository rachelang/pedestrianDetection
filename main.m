% setup
clear; close all; clc
addpath('./data'); addpath('./lib');

% neural network specifications
num_hidden_layers = 1;  % will be varied to get best result
input_layer_size = 3024;
hidden_layer_size = 50;
num_labels = 1;

nn_specs = [num_hidden_layers, input_layer_size, hidden_layer_size, num_labels];

% load training and testing data
X_train_all = [importdata('pedTrain.mat'); importdata('nonpedTrain.mat')];
y_train_all = [ones(52112, 1); zeros(32465, 1)];
X_test_all = [importdata('pedTest.mat'); importdata('nonpedTest.mat')];
y_test_all = [ones(25608, 1); zeros(16235, 1)];

% partition data (because of personal slow machine,
% only chose some of dataset)
rand_train = randperm(size(X_train_all, 1));
X_train = X_train_all(rand_train(1:40000), :);
y_train = y_train_all(rand_train(1:40000), :);

halfTest = round(size(X_test_all, 1) / 2);
rand_cv = randperm(halfTest); % half of test set -> cv
X_cv = X_test_all(rand_cv(1:8000), :);
y_cv = y_test_all(rand_cv(1:8000), :);

rand_test = randperm(size(X_test_all, 1) - halfTest) + halfTest; % half of test set -> test
X_test = X_test_all(rand_cv(1:8000), :);
y_test = y_test_all(rand_cv(1:8000), :);

% visualize data
% displayData(reshape(X_train(123, :), 84, 36));

% Neural Network Training
% -----------------------

% training network using fmincg rather than the native fminunc, as this 
% optimization algorithm uses much less memory and makes it possible 
% to be run on older machines (like mine)
max_iters = 40;

% vary lambda to find best result
lambda_vec = 0.01; %[0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];  
[ThetaRolled, lambda_train_cost, lambda_cv_cost, bestLambda] = varyLambda(lambda_vec, ...
                                              max_iters, nn_specs, ...
                                              X_train, y_train, ...
                                              X_cv, y_cv);

Theta = reshapeParams(ThetaRolled, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels);                                          
y_pred = predict(Theta, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(y_pred == y_test)) * 100);
[test_cost, ~] = costFunction(ThetaRolled, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_test, y_test, bestLambda);
fprintf('\nTest Set error: %f\n', test_cost);
                                             
% save trained theta
save theta.mat Theta;

% plot errors as function of lambda                                             
figure;
plot(lambda_vec, lambda_train_cost, lambda_vec, lambda_cv_cost);
title('Error as function of lambda')
legend('Train', 'Cross Validation')
xlabel('Lambda')
ylabel('Error')

%Theta = importdata('theta.mat');

% load detection test image
test_img = bmpToMatrix('sidewalk_242.bmp');
rescale(test_img, 3);
scale = 3;
[img_h, img_w] = size(test_img);
min_img_h = round(img_h*0.1);

% Sliding Window algorithm
border_img = test_img;
start_x = 1; start_y = 1;
h_incr = max(round(84*0.1), 1);
w_incr = max(round(36*0.1), 1);

fprintf('Starting sliding window detection: pyramid layers\n');
a = 0; %%%%%%%%%%
while(img_h > min_img_h)
    while(start_y + 84 - 1 <= img_h)
        while(start_x + 36 - 1 <= img_w)
            window = test_img(start_y:start_y + 84 - 1, start_x:start_x + 36 - 1);
            y_pred = predict(Theta, rollParameters(window));
            
            if(y_pred == 1)
                a = a+1;
                border_img = drawBorder(border_img, start_x, start_y, ...
                                        84, 36);
            end
            start_x = start_x + w_incr;
        end
        start_x = 1;
        start_y = start_y + h_incr;
    end
    start_y = 1;
    test_img = rescale(test_img, 0.9);
    scale = scale*0.9;
    [img_h, img_w] = size(test_img);
end

a
displayData(border_img);