function [sigma_tex, sigma_smooth] = learn_sigma_tex(w, config)
%LEARN_SIGMA_TEX

% create filter banks
cur_index = 1;
d = 2; % dimension of filters
num_possible_filts = 5;
filter_banks = cell(1,num_possible_filts);

if config.use_serge_filts
    filter_banks{cur_index} = FbMake(d, 1, config.show);
    cur_index = cur_index+1;
end
if config.use_doog_filts
    filter_banks{cur_index} = FbMake(d, 2, config.show);
    cur_index = cur_index+1;
end
if config.use_LM_filts
    filter_banks{cur_index} = makeLMfilters(config.filt_size);
    cur_index = cur_index+1;
end
if config.use_LW_filts
    filter_banks{cur_index} = makeLWfilters();
    cur_index = cur_index+1;
end
filter_banks = filter_banks(1:(cur_index-1));

num_training = size(config.training_nums, 1);
D_mse = zeros(num_training, 1);

depth_weight_count = 0.0;
sigma_smooth = 0.0;

% loop through training set and predict each with a linear function 
for k = 1:num_training
    % read the next filename
    img_num = config.training_nums(k);
    rgb_filename = sprintf(config.rgb_file_template, img_num);
    depth_filename = sprintf(config.depth_file_template, img_num);
    
    if mod(k-1, config.out_rate) == 0
        fprintf('Computing tex error for image %d: %s\n', k, rgb_filename);
    end
    
    % load an RGBD pair
    image_pyr = load_image_pyramid(rgb_filename, depth_filename, config);
    [Phi, I_gradients] = ...
        extract_texture_features(image_pyr, filter_banks, config);
    [~, D_diff] = ...
        create_depth_diff_weights(image_pyr, I_gradients, config);
    
    % form target depth_vector
    D_vec = double(image_pyr.D_pyr{1});
    D_vec = D_vec(:);
    D_target_vec = D_vec;
    if config.use_inv_depth
       D_target_vec = config.max_depth ./ D_vec;
    end
    if config.use_log_depth
       D_target_vec = log(D_vec+1);
    end
 
    % update max-likelihood tex depth error
    D_pred_vec = Phi * w;
    D_sq_error = (D_pred_vec - D_target_vec).^2;
    D_mse(k) = mean(D_sq_error);
    
    % update max-likelihood smoothness param
    height = image_pyr.im_height;
    width = image_pyr.im_width;
    num_pix = height*width;
    sigma_smooth = sigma_smooth + sum(sum(D_diff{1}));
    sigma_smooth = sigma_smooth + sum(sum(D_diff{2}));
    sigma_smooth = sigma_smooth + sum(sum(D_diff{3}));
    sigma_smooth = sigma_smooth + sum(sum(D_diff{4}));
    depth_weight_count = depth_weight_count + 4 * num_pix;
    
    if isnan(sigma_smooth)
        stop = 1; 
    end
end
% ML tex estimate
sigma_tex = mean(D_mse);
% ML sigma smooth estimate
sigma_smooth = sigma_smooth / depth_weight_count;


end

