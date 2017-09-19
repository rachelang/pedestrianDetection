function rolled_params = rollParameters(params)

n = size(params, 2);
rolled_params = [];

for i = 1:n
    rolled_params = [rolled_params; params(:, i)];
end

rolled_params = rolled_params';

end