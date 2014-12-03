function costs = tps_objective(tps, y, y_t_true)
%TPS_OBJECTIVE Evaluate tps objective

costs = zeros(3,1);

n = size(tps.x, 1);
d = size(tps.x, 2);

y_transformed = tps_apply(tps, y);
bend_coefs = tps.bend_coef * ones(d, 1);
w = repmat(tps.weights, 1, d);
K_nn = tps_kernel_matrix(tps.x);

% matching
costs(1) = sum(sum(w .* (y_t_true - y_transformed).^2));

% bending
costs(2) = trace(diag(bend_coefs) * tps.w_ng' * K_nn * tps.w_ng);

% rotation
A = tps.lin_ag - eye(d);
costs(3) = trace(A' * diag(tps.rot_coef) * A);

end

