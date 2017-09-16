% setup
clear; close all; clc
addpath('./data');

% neural network specifications
num_hidden_layers = 3;  % will be varied to get best result
input_layer_size = 784;
hidden_layer_size = 10;
num_labels = 10;

nn_specs = [num_hidden_layers, input_layer_size, hidden_layer_size, num_labels];

% load training and testing data
pedTrain = importdata('pedTrain.mat');
nonpedTrain = importdata('nonpedTrain.mat');
pedTest = importdata('pedTest.mat');
nonpedTest = importdata('nonpedTest.mat');

X_train = [pedTrain(1:30000, :); nonpedTrain(1:10000, :)];
y_train = [ones(30000, 1); zeros(10000, 1)];

X_cv = [pedTest(1:5000, :); nonpedTest(1:2000, :)];
y_cv = [ones(5000, 1); zeros(2000, 1)];

X_test = [pedTest(5001:10000, :); nonpedTest(2001:4000, :)];
y_test = [ones(5000, 1); zeros(2000, 1)];

% constant(s)
m_train = size(X_train, 1);

% visualize data


% Neural Network Training
% -----------------------

% training network using fmincg rather than the native fminunc, as this 
% optimization algorithm uses much less memory and makes it possible 
% to be run on older machines (like mine)
max_iters = 100;

% vary lambda to find best result
lambda_vec = 0; %[0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];  
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