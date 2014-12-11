% simple test to check whether tps is working in 3d

n = 100;
d = 3;

vis = true;
x = 10*rand(n, d);

%% rotation test
rng(100);
R = rand(3);
[U, S, V] = svd(R);
R = U * V';

A = 0.001*rand(n, d);
B = [R rand(d, 1)];
K = tps_kernel_matrix(x);

y = A' * K + B * [x ones(n,1)]';
y = y';

bend_coef = 0.1;
rot_coef = 100000;
n_fit = n / 2;
x_fit = x(1:n_fit,:);
y_fit = y(1:n_fit,:);
w = ones(n_fit,1);
tps = tps_fit(x_fit, y_fit, bend_coef, rot_coef, w);

x_t = tps_apply(tps, x);
costs = tps_objective(tps, x_fit, y_fit);

y_diff = y - x_t;
y_diff_orig = y - x;

if vis
    figure(1);
    scatter3(x(:,1), x(:,2), x(:,3), 'r');
    hold on;
    scatter3(y(:,1), y(:,2), y(:,3), 'g');
    scatter3(x_t(:,1), x_t(:,2), x_t(:,3), 'b');
end

fprintf('\nRotation test case\n');
fprintf('Warping cost:\t %f\n', costs(1));
fprintf('Bending cost:\t %f\n', costs(2));
fprintf('Rotation cost:\t %f\n', costs(3));


%% affine test
rng(200);
A = 0.001*rand(n, d);
B = rand(d, d+1);
K = tps_kernel_matrix(x);

y = A' * K + B * [x ones(n,1)]';
y = y';

bend_coef = 0.1;
rot_coef = 0;
w = ones(n,1);
tps = tps_fit(x, y, bend_coef, rot_coef, w);

x_t = tps_apply(tps, x);
costs = tps_objective(tps, x, y);

y_diff = y - x_t;

if vis
    figure(1);
    scatter3(x(:,1), x(:,2), x(:,3), 'r');
    hold on;
    scatter3(y(:,1), y(:,2), y(:,3), 'g');
    scatter3(y_t(:,1), y_t(:,2), y_t(:,3), 'b');
end

fprintf('\nAffine test case\n');
fprintf('Warping cost:\t %f\n', costs(1));
fprintf('Bending cost:\t %f\n', costs(2));
fprintf('Rotation cost:\t %f\n', costs(3));
