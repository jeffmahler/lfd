function [d_pred_vec, varargout] = predict_depth_gaussian_tps(image_pyr, ...
    Phi, corr_ind, u_nc, v_3d, smooth_weights, tps_weights, model, config)
%PREDICT_DEPTH_GAUSSIAN Predicts depth using gaussian model of both depth
%  and smoothness, which can be solved as a linear system
% u_nc - test pixels in normalized image coords
% v_3d - predicted test points in 3d

% lambda = model.sigma_tex / model.sigma_smooth;
% gamma = model.sigma_tex / model.sigma_prior;
% nu = model.sigma_tex / model.sigma_tps;

% temp weights matrix organized as (L R U D)
height = image_pyr.im_height;
width = image_pyr.im_width;
num_pix = height * width;
A = zeros(num_pix);

% predict texture depth with row-wise models
im_size = [image_pyr.im_height, image_pyr.im_width];
num_pix = im_size(1) * im_size(2);
lin_ind = 1:num_pix;
[row_ind, ~] = ind2sub(im_size, lin_ind);
b = zeros(num_pix, 1);
vert_model_size = ceil(im_size(1) / config.num_vert_models);

% relative weights
lambda = zeros(num_pix,1);
gamma = zeros(num_pix,1);
nu = zeros(num_pix,1);

% predict depth per-row
start_row = 1;
end_row = min(start_row + vert_model_size, im_size(1)+1);
for j = 1:config.num_vert_models
    v_row_ind = row_ind >= start_row & row_ind < end_row;
    v_lin_ind = lin_ind(v_row_ind);
    Phi_v = Phi(v_lin_ind,:);

    b(v_lin_ind) = Phi_v * model.w{j};
    lambda(v_lin_ind) = model.sigma_tex{j} / model.sigma_smooth;
    gamma(v_lin_ind) = model.sigma_tex{j} / model.sigma_prior;
    nu(v_lin_ind) = model.sigma_tex{j} / model.sigma_tps;

    start_row = start_row + vert_model_size;
    end_row = min(start_row + vert_model_size, im_size(1)+1);
end
%b = Phi * model.w;

% get the correspondences
target_lin_ind = corr_ind(:,1);
source_lin_ind = corr_ind(:,2);
corr_weights = zeros(num_pix, 1);
corr_weights(target_lin_ind) = tps_weights(target_lin_ind);

% compute the sums needed for prediction using tps
u_nc_sq_sum = zeros(num_pix, 1);
uv_sum = zeros(num_pix, 1);
u_nc_sq_sum(target_lin_ind) = sum(u_nc(target_lin_ind,:).^2, 2);
uv_sum(target_lin_ind) = sum(u_nc(target_lin_ind,:) .* v_3d(source_lin_ind,:), 2);

% create rhs of equation
b = b + gamma .* model.D_prior_vec;
b = b + nu .* corr_weights .* uv_sum;

% if config.use_log_depth
%     b = exp(b) - 1;
% end

%weights = zeros(num_pix, 4);

% update off-diagonal
im_size = [height width];
lin_ind = 1:num_pix;
mat_size = size(A);

% create indices to left and right of current pixel
[px_y, px_x] = ind2sub(im_size, lin_ind);
one_vec = ones(1, num_pix);
ind_left =  [max([px_x - 1; one_vec]);
             px_y];
ind_right = [min([px_x + 1; width*one_vec]);
             px_y];
ind_up =    [px_x;
             max([px_y - 1; one_vec])];
ind_down =  [px_x;
             min([px_y + 1; height*one_vec])];

% get valid indices
valid_ind_left = find(px_x - 1 > 0);
valid_ind_right = find(px_x + 1 <= width);
valid_ind_up = find(px_y - 1 > 0);
valid_ind_down = find(px_y + 1 <= height);
         
% compute linear indices
lin_ind_left = sub2ind(im_size, ind_left(2,:), ind_left(1,:));
lin_ind_right = sub2ind(im_size, ind_right(2,:), ind_right(1,:));
lin_ind_up = sub2ind(im_size, ind_up(2,:), ind_up(1,:));
lin_ind_down = sub2ind(im_size, ind_down(2,:), ind_down(1,:));
lin_ind_diag = 1:(num_pix+1):(num_pix*num_pix);

% fill left symmetric
lin_ind_fill = sub2ind(mat_size, lin_ind, lin_ind_left);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda .* smooth_weights(:,1);
lin_ind_fill = sub2ind(mat_size, lin_ind_left, lin_ind);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda .* smooth_weights(:,1);

% fill right symmetric
lin_ind_fill = sub2ind(mat_size, lin_ind, lin_ind_right);
%A(lin_ind_fill) = -2 * lambda * weights(:,2);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda .* smooth_weights(:,2);
lin_ind_fill = sub2ind(mat_size, lin_ind_right, lin_ind);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda .* smooth_weights(:,2);

% fill up symmetric
lin_ind_fill = sub2ind(mat_size, lin_ind, lin_ind_up);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda .* smooth_weights(:,3);
lin_ind_fill = sub2ind(mat_size, lin_ind_up, lin_ind);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda .* smooth_weights(:,3);

% fill down symmetric
lin_ind_fill = sub2ind(mat_size, lin_ind, lin_ind_down);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda .* smooth_weights(:,4);
lin_ind_fill = sub2ind(mat_size, lin_ind_down, lin_ind);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda .* smooth_weights(:,4);

% update diagonal
A(lin_ind_diag) = ...
    ones(num_pix,1) + lambda .* sum(smooth_weights, 2) + gamma + ...
    nu .* corr_weights .* u_nc_sq_sum; % corr weights zeros out irrelevant corrs

lin_ind_fill = ...
    sub2ind(mat_size, lin_ind_left(valid_ind_left), lin_ind_left(valid_ind_left));
A(lin_ind_fill) = A(lin_ind_fill)' + ...
    lambda(lin_ind(valid_ind_left)) .* smooth_weights(lin_ind(valid_ind_left),1); % edge right of left neighbor

lin_ind_fill = ...
    sub2ind(mat_size, lin_ind_right(valid_ind_right), lin_ind_right(valid_ind_right));
A(lin_ind_fill) = A(lin_ind_fill)' + ...
    lambda(lin_ind(valid_ind_right)) .* smooth_weights(lin_ind(valid_ind_right),2); % edge left of right neighbor

lin_ind_fill = ...
    sub2ind(mat_size, lin_ind_up(valid_ind_up), lin_ind_up(valid_ind_up));
A(lin_ind_fill) = A(lin_ind_fill)' + ...
    lambda(lin_ind(valid_ind_up)) .* smooth_weights(lin_ind(valid_ind_up),3); % edge below upper neighbor

lin_ind_fill = ...
    sub2ind(mat_size, lin_ind_down(valid_ind_down), lin_ind_down(valid_ind_down));
A(lin_ind_fill) = A(lin_ind_fill)' + ...
    lambda(lin_ind(valid_ind_down)) .* smooth_weights(lin_ind(valid_ind_down),4); % edge above upper neighbor

% solve linear system
M = sparse(A);
%d_pred_vec = pcg(M, b, config.tol, config.max_iters);
d_pred_vec = bicg(M, b, config.tol, config.max_iters);

if nargout > 1
    varargout{1} = A;
end
if nargout > 1
    varargout{2} = b;
end

end

