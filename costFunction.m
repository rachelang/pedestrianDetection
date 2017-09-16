function [J, grad] = costFunction(params, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
    
% Constants
num_layers = num_hidden_layers + 2;
m = size(X, 1);

% Reroll parameters into weights 
Theta = reshapeParams(params, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels);

% Forward propagation
a = cell(num_layers, 1);
z = cell(num_layers, 1);
a{1} = [ones(m,1) X]';

for i = 1:(num_layers - 1)
    z{i + 1} = Theta{i} * a{i};
    a{i + 1} = sigmoid(z{i + 1});
    num_units = size(a{i + 1}, 2);
    if i ~= (num_layers - 1)
        a{i + 1} = [ones(1, num_units); a{i + 1}];
    end
end

% Hypothesis result in vector form
h = a{num_layers}';
% Labels in vector form
yv = 0:(num_labels - 1) == y;

% Cost function cost term
cost = -yv.*log(h) - (1-yv).*log(1-h);

% Cost function regression term
Theta_reg = cell(num_layers - 1, 1);
regSum = 0;
for i = 1:(num_layers - 1)
    Theta_reg{i} = Theta{i}(:, 2:size(Theta{i}, 2));
    temp = Theta_reg{i} .^ 2;
    regSum = regSum + sum(temp(:));
end

% Cost function
J = 1/m*sum(cost(:))+lambda/(2*m)*regSum;

% Backpropagation for determining gradient
e = cell(num_layers, 1);
e{num_layers} = a{num_layers} - yv';

for i = (num_layers - 1):-1:2
    temp = Theta{i}' * e{i + 1};
    temp(1, :) = []; % remove bias unit
    e{i} = temp .* sigmoidGradient(z{i});
end

% Initialize zero for elements in theta delta for adding over examples
Theta_delta = cell(num_layers - 1, 1);
for i = 1:(num_layers - 1)
    Theta_delta{i} = zeros(size(Theta{i}));
end

% Update theta delta terms over all examples
for i = 1:m
    for j = 1:(num_layers - 1)
        Theta_delta{j} = Theta_delta{j} + e{j + 1}(:, i) * a{j}(:, i)';
    end
end

% Calculate theta regularized term
for i = 1:(num_layers - 1)
    Theta_reg{i} = [zeros(size(Theta{i}, 1), 1) Theta_reg{i}];
end

% Calculate theta gradient using theta delta and theta regularized terms
Theta_grad = cell(num_layers - 1, 1);
for i = 1:(num_layers - 1)
    Theta_grad{i} = 1/m * Theta_delta{i} + lambda/m * Theta_reg{i};
end

% Unroll gradients for advanced optimization algorithms
grad = zeros(size(params));
index = 1;

for i = 1:(num_layers - 1)
    grad(index:(index + numel(Theta_grad{i}) - 1)) = Theta_grad{i}(:); 
    index = index + numel(Theta_grad{i});
end

end