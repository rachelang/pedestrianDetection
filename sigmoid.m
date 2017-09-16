function g = sigmoid(z)
% Calculates elementwise sigmoid function
% z can be vector or matrix

g = 1.0 ./ (1.0 + exp(-z));

end
