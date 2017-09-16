function Theta = reshapeParams(params, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels)
% Reroll parameters into weights for neural network with the same number
% of nodes for every hidden layer
num_layers = num_hidden_layers + 2;

Theta = cell(num_layers - 1, 1);

index = hidden_layer_size * (input_layer_size + 1);
Theta{1} = reshape(params(1:index), hidden_layer_size, (input_layer_size + 1));
             
hidden_theta_length = hidden_layer_size * (hidden_layer_size + 1);

for i = 2:num_hidden_layers
    Theta{i} = reshape(params((index + 1):(index + hidden_theta_length)), ...
                 hidden_layer_size, (hidden_layer_size + 1));
    index = index + hidden_theta_length;
end

Theta{num_layers - 1} = reshape(params((index + 1):end), ...
                 num_labels, (hidden_layer_size + 1));
end