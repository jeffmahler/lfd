function [weights, D_diffs_weighted] = ...
    create_texture_diff_weights(image_pyr, I_filt_resp, num_filts, config)
%EXTRACT_DEPTH_DIFF_WEIGHTS Creates both image intensity weights and
%   computes depth difference for monocular depth smoothness regularization

% hardcoded depth diff filters
diff_right_filter = ...
    [0, 0, 0;
    -1, 1, 0;
    0, 0, 0];
diff_left_filter = ...
    [0, 0, 0;
    0, 1,-1;
    0, 0, 0];
diff_down_filter = ...
    [0, -1, 0;
    0, 1, 0;
    0, 0, 0];
diff_up_filter = ...
    [0, 0, 0;
    0, 1, 0;
    0, -1, 0];

num_neighbors = 4;
base_level = 1;

grad_scale = config.grad_scale;
use_inv_depth = config.use_inv_depth;
use_log_depth = config.use_log_depth;

height = image_pyr.im_height;
width = image_pyr.im_width;
num_pix = height * width;
num_features = size(I_filt_resp, 2);

weights = zeros(num_pix, num_neighbors);
D_diff = double(image_pyr.D_pyr{base_level});

if use_inv_depth
   D_diff = config.max_depth ./ (D_diff + 1); 
end
if use_log_depth
   D_diff = log(D_diff+1); 
end

% compute texture gradients
tex_resp_im = reshape(I_filt_resp, height, width, num_features);
tex_abs_diff_x = zeros(height, width);
tex_abs_diff_y = zeros(height, width);
for i = 1:num_filts
    diff_tex_left = conv2(tex_resp_im(:,:,i), diff_left_filter, 'same');
    diff_tex_right = conv2(tex_resp_im(:,:,i), diff_right_filter, 'same');
    tex_abs_diff_x = tex_abs_diff_x + abs(diff_tex_left) + abs(diff_tex_right);
    
    diff_tex_up = conv2(tex_resp_im(:,:,i), diff_up_filter, 'same');
    diff_tex_down = conv2(tex_resp_im(:,:,i), diff_down_filter, 'same');
    tex_abs_diff_y = tex_abs_diff_y + abs(diff_tex_up) + abs(diff_tex_down);
end
tex_abs_diff_x = tex_abs_diff_x / (2 * num_filts);
tex_abs_diff_y = tex_abs_diff_y / (2 * num_filts);

% compute weights based on image color difference (could also be texture)
diff_D_weights_x = 1.0 ./ (1.0 + exp(grad_scale * tex_abs_diff_x));
diff_D_weights_y = 1.0 ./ (1.0 + exp(grad_scale * tex_abs_diff_y));
weights(:,1) = diff_D_weights_x(:);
weights(:,2) = diff_D_weights_x(:);
weights(:,3) = diff_D_weights_y(:);
weights(:,4) = diff_D_weights_y(:);
weights(weights < 1e-4) = 0;

% zero out border weights
im_size = [height width];
lin_ind_left = sub2ind(im_size, 1:height, ones(1,height));
weights(lin_ind_left, 1) = 0;
lin_ind_right = sub2ind(im_size, 1:height, width*ones(1,height));
weights(lin_ind_right, 2) = 0;
lin_ind_top = sub2ind(im_size, ones(1,width), 1:width);
weights(lin_ind_top, 3) = 0;
lin_ind_bot = sub2ind(im_size, height*ones(1,width), 1:width);
weights(lin_ind_bot, 4) = 0;

% create depth differences
diff_D_left = conv2(D_diff, diff_left_filter, 'same');
diff_D_right = conv2(D_diff, diff_right_filter, 'same');
diff_D_up = conv2(D_diff, diff_up_filter, 'same');
diff_D_down = conv2(D_diff, diff_down_filter, 'same');

% weight the depth differences
D_diffs_weighted = cell(1, num_neighbors);
D_diffs_weighted{1} = diff_D_weights_x .* diff_D_left.^2;
D_diffs_weighted{2} = diff_D_weights_x .* diff_D_right.^2;
D_diffs_weighted{3} = diff_D_weights_y .* diff_D_up.^2;
D_diffs_weighted{4} = diff_D_weights_y .* diff_D_down.^2;

end

