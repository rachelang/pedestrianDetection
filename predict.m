function p = predict(Theta, X)
% Predicts the label of an input given weights trained by neural network

% Constants
m = size(X, 1);
num_layers = length(Theta) + 1;

% Forward propagation (simple version)
a = [ones(m,1) X]';

for i = 1:(num_layers - 1)
    z = Theta{i} * a;
    a = sigmoid(z);
    num_units = size(a, 2);
    if i ~= (num_layers - 1)
        a = [ones(1, num_units); a];
    end
end

h = a';

[~, p] = max(h, [], 2);
p = p - 1;

end