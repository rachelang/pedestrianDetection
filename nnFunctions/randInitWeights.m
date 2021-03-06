function W = randInitWeights(num_hidden_layers, ...
                               input_layer_size, ...
                               hidden_layer_size, ...
                               num_labels)
% Initialize (unrolled) weights to random value [-epsilon_init, epsilon_init] in layer that has 
% L_in incoming connections and L_out outgoing connections to break symmetry
params_length = (input_layer_size + 1) * hidden_layer_size ...
                + (num_hidden_layers - 1) * (hidden_layer_size + 1) * hidden_layer_size ...
                + (hidden_layer_size + 1) * num_labels;
epsilon_init = 0.12;
W = rand(params_length, 1) * 2 * epsilon_init - epsilon_init;

end