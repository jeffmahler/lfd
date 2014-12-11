function thin_plate_spline = tps_fit(x, y, bend_coef, rot_coef, w)
%TPS_FIT Fit a thin-plate-spline to the data specified in x and y
%   using smoothing coefficients bend_coef and rot_coef

num_source_pts = size(x, 1);
num_target_pts = size(y, 1);
n = min(num_source_pts, num_target_pts);
d = size(x, 2);

if num_source_pts ~= num_target_pts
   fprintf('Warning: Different number of source and target points. Defaulting to the smaller of the two.\n'); 
end

if nargin < 5
   w = ones(n,1);
end

K_nn = tps_kernel_matrix(x);
Q = [ones(n, 1), x, K_nn];

if isscalar(rot_coef)
    rot_coefs = ones(d,1) * rot_coef;
else
    rot_coefs = rot_coef;
end

x = x(1:n,:);
y = y(1:n,:);
A = [zeros(d+1); ones(n,1) x]';

% atm default to solving all dims at once
WQ = diag(w) * Q;
QWQ = Q' * WQ;

H = QWQ;
H(d+2:end, d+2:end) = H(d+2:end, d+2:end) + bend_coef * K_nn;
H(2:d+1, 2:d+1) = H(2:d+1, 2:d+1) + diag(rot_coefs);

f = -WQ' * y;
f(2:d+1, 1:d) = f(2:d+1, 1:d) - diag(rot_coefs);

% solve eqp1 from python code
n_vars = size(H, 1);
n_cnts = size(A, 1);
[U, S, V] = svd(A');

% null space
N = U(:, n_cnts+1:end);
L = N' * H * N;
R = -N' * f;
z = L \ R;
theta = N * z;

thin_plate_spline = struct();
thin_plate_spline.trans_g = theta(1,:);
thin_plate_spline.lin_ag = theta(2:d+1,:);
thin_plate_spline.w_ng = theta(d+2:end,:);
thin_plate_spline.x = x;
thin_plate_spline.weights = w;
thin_plate_spline.rot_coef = rot_coefs;
thin_plate_spline.bend_coef = bend_coef;

end

