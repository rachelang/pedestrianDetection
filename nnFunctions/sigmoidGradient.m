function g = sigmoidGradient(z)
% Calculates the elementwise gradient of the sigmoid function evaluated at z
% z can be vector or matrix

g = sigmoid(z).*(1-sigmoid(z));

end
