function X_trim = trimBorder(X, h, w, p)

m = size(X, 1);
X_trim = zeros(m, (h - 2*p)*(w - 2*p));

for i = 1:m
    x_trim = X(i, :);
    
    x_trim((h*w - h*p + 1):(h*w)) = []; % trim right border

    for col = (w-p):-1:(p + 1)
        x_trim((col*h - p + 1):(col*h)) = []; % trim bottom border
        x_trim(((col - 1)*h + 1):((col - 1)*h + p)) = []; % trim top border
    end

    x_trim(1:(h*p)) = []; % trim left border

    X_trim(i, :) = x_trim;
end

end
