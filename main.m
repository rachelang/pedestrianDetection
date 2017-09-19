% setup
clear; close all; clc
addpath('./data'); addpath('./lib'); addpath('./sampleImages'); addpath('./trainedTheta');
addpath('./helpers'); addpath('./nnFunctions');

% neural network specifications
num_hidden_layers = 1;  % will be varied to get best result
input_layer_size = 3024;
hidden_layer_size = 150;
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
X_train = X_train_all;
y_train = y_train_all;

halfTest = round(size(X_test_all, 1) / 2);
rand_cv = randperm(halfTest); % half of test set -> cv
X_cv = X_test_all(rand_cv, :);
y_cv = y_test_all(rand_cv, :);

rand_test = randperm(size(X_test_all, 1) - halfTest) + halfTest; % half of test set -> test
X_test = X_test_all(rand_cv, :);
y_test = y_test_all(rand_cv, :);

% Neural Network Training
% -----------------------

% training network using fmincg rather than the native fminunc, as this 
% optimization algorithm uses much less memory and makes it possible 
% to be run on older machines (like mine)
max_iters = 40;

% vary lambda to find best result
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];  
[ThetaRolled, lambda_train_cost, lambda_cv_cost, bestLambda] = varyLambda(lambda_vec, ...
                                              max_iters, nn_specs, ...
                                              X_train, y_train, ...
                                              X_cv, y_cv);

Theta = reshapeParams(ThetaRolled, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels); 

% Try using pre-trained theta to see results on real data:
% Theta = importdata('1layer150units');

y_pred = predict(Theta, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(y_pred == y_test)) * 100); %%%%%%%

[test_cost, ~] = costFunction(ThetaRolled, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_test, y_test, bestLambda);

fprintf('\nTest Set error: %f\n', test_cost);

% save trained theta
save('theta.mat', 'Theta');

% plot errors as function of lambda                                             
figure;
plot(lambda_vec, lambda_train_cost, lambda_vec, lambda_cv_cost);
title('Error as function of lambda')
legend('Train', 'Cross Validation')
xlabel('Lambda')
ylabel('Error')

% detection constants
scale = 2;
inc_percent = 0.3;
min_img_percent = 0.3;

% load detection test image
test_img = bmpToMatrix('walkfeet_27.bmp');
border_img = test_img;
test_img = rescale(test_img, scale);
[img_h, img_w] = size(test_img);
min_img_h = round(img_h*min_img_percent);

% Sliding Window algorithm
start_x = 1; start_y = 1;
h_incr = max(round(84*inc_percent), 1);
w_incr = max(round(36*inc_percent), 1);
num_boxes = 0;

fprintf('Sliding Window Detection: pyramid layers\n');
fprintf('----------------------------------------\n');

while(img_h > min_img_h)
    while(start_y + 84 - 1 <= img_h)
        while(start_x + 36 - 1 <= img_w)
            window = test_img(start_y:start_y + 84 - 1, start_x:start_x + 36 - 1);
            y_pred = predict(Theta, rollParameters(window));
            
            if(y_pred == 1)
                num_boxes = num_boxes + 1;
                border_img = drawBorder(border_img, max(floor(start_x/scale), 1), max(floor(start_y/scale), 1), ...
                                        floor(84/scale), floor(36/scale));
            end
            start_x = start_x + w_incr;
        end
        start_x = 1;
        start_y = start_y + h_incr;
    end
    start_y = 1;
    
    test_img = rescale(test_img, 0.9);
    [img_h, img_w] = size(test_img);
    scale = scale*0.9;
end

% Final detection image
displayData(border_img);
fprintf('Number of boxes (pos): %i\n', num_boxes);