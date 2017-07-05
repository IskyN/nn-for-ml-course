function predictions = predict(W1, W2, X)
% W1: (n x m) matrix -- input-hidden weights
% W2: m-vector -- hidden-output weights
% X:  (k x n) matrix -- data points (sets of inputs)
    
    n = rows(W1);
    m = columns(W1);
    k = rows(X);
    
    % Check that parameter sizes match up 
    assert (columns(W2) == 1, "W2 not a column vector")
    assert (rows(W2) == m, "W1 width and W2 height mismatched")
    assert (columns(X) == n, "X width and W1 height mismatched")

    % One pass through the NN for each data point (online)
    H = zeros(n, m);
    for d = 1:k,
        H = X[:,d] .* W1;  % hidden unit values

endfunction

