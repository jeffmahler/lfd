function K = tps_kernel_matrix(x, y)
%TPS_KERNEL_MATRIX Compute tps kernel matrix between x and y

dim = size(x, 2);
if nargin < 2
   y = x; 
end

sq_dist_mat = sq_dist(x', y');

if dim == 2
    K = 4 * sq_dist_mat * log(sqrt(sq_dist_mat) + 1e-20);
elseif dim == 3
    K = -sqrt(sq_dist_mat);
else
    fprintf('Warning: Unknown TPS kernel for dimension %f\n', dim);
    K = sq_dist_mat; 
end

end

