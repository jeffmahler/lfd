function [sigma_tex, sigma_smooth, sigma_prior] = learn_sigma_tex(model, config)
%LEARN_SIGMA_TEX

% create filter banks
cur_index = 1;
d = 2; % dimension of filters
num_possible_filts = 5;
filter_banks = cell(1,num_possible_filts);
num_filters = 0;

if config.use_serge_filts
    filter_banks{cur_index} = FbMake(d, 1, config.show);
    num_filters = num_filters + size(filter_banks{cur_index}, 3);
    cur_index = cur_index+1;
end
if config.use_doog_filts
    filter_banks{cur_index} = FbMake(d, 2, config.show);
    num_filters = num_filters + size(filter_banks{cur_index}, 3);
    cur_index = cur_index+1;
end
if config.use_LM_filts
    filter_banks{cur_index} = makeLMfilters(config.filt_size);
    num_filters = num_filters + size(filter_banks{cur_index}, 3);
    cur_index = cur_index+1;
end
if config.use_LW_filts
    filter_banks{cur_index} = makeLWfilters();
    num_filters = num_filters + size(filter_banks{cur_index}, 3);
    cur_index = cur_index+1;
end
filter_banks = filter_banks(1:(cur_index-1));

num_training = size(config.training_nums, 1);
%D_tex_mse = zeros(num_training, 1);
D_prior_mse = zeros(num_training, 1);

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
         create_texture_diff_weights(image_pyr, Phi, num_filters, config);
%    [~, D_diff] = ...
%        create_depth_diff_weights(image_pyr, I_gradients, config);
    
    % form target depth_vector
    D_vec = double(image_pyr.D_pyr{1});
    D_vec = D_vec(:);
    D_target_vec = D_vec;
    if config.use_inv_depth
       D_target_vec = config.max_depth ./ (D_vec + 1);
    end
    if config.use_log_depth
       D_target_vec = log(D_vec+1);
    end
 
    % update max-likelihood tex depth error
%    D_pred_vec = Phi * model.w{1};
    im_size = [image_pyr.im_height, image_pyr.im_width];
    num_pix = im_size(1) * im_size(2);
    lin_ind = 1:num_pix;
    [row_ind, ~] = ind2sub(im_size, lin_ind);
    D_pred_vec = zeros(num_pix, 1);
    vert_model_size = ceil(im_size(1) / config.num_vert_models);
        
    % predict depth per-row
    start_row = 1;
    end_row = min(start_row + vert_model_size, im_size(1)+1);
    for j = 1:config.num_vert_models
        v_row_ind = row_ind >= start_row & row_ind < end_row;
        v_lin_ind = lin_ind(v_row_ind);
        Phi_v = Phi(v_lin_ind,:);
        
        D_pred_vec(v_lin_ind) = Phi_v * model.w{j};
        
        start_row = start_row + vert_model_size;
        end_row = min(start_row + vert_model_size, im_size(1)+1);
    end
    
%     D_im = reshape(exp(D_pred_vec)-1, im_size);
%     figure(13);
%     imshow(histeq(uint16(D_im)));
%     hold on;
%     for j = 1:config.num_vert_models
%         plot([0;im_size(2)], [vert_model_size*(j-1); vert_model_size*(j-1)]);
%     end
%     pause(10);
    
    D_sq_error = (D_pred_vec - D_target_vec).^2;
    if ~exist('D_tex_mse', 'var')
        D_tex_mse = zeros(num_pix, 1);
    end
    D_tex_mse = D_tex_mse + D_sq_error;
    
    % update max-likelihood smoothness param
    height = image_pyr.im_height;
    width = image_pyr.im_width;
    num_pix = height*width;
    sigma_smooth = sigma_smooth + sum(sum(D_diff{1}));
    sigma_smooth = sigma_smooth + sum(sum(D_diff{2}));
    sigma_smooth = sigma_smooth + sum(sum(D_diff{3}));
    sigma_smooth = sigma_smooth + sum(sum(D_diff{4}));
    depth_weight_count = depth_weight_count + 4 * num_pix;
    
    % update max-likelihood prior
    D_prior_sq_error = (model.D_prior_vec - D_target_vec).^2;
    D_prior_mse = mean(D_prior_sq_error);
    
    if isnan(sigma_smooth)
        stop = 1; 
    end
end
% ML tex estimate
sigma_tex = cell(1, config.num_vert_models);
start_row = 1;
end_row = min(start_row + vert_model_size, im_size(1)+1);
for j = 1:config.num_vert_models
    v_row_ind = row_ind >= start_row & row_ind < end_row;
    v_lin_ind = lin_ind(v_row_ind);
    
    sigma_tex{j} = sum(D_tex_mse(v_lin_ind)) / ...
        (num_training * size(v_lin_ind,2));
    
    start_row = start_row + vert_model_size;
    end_row = min(start_row + vert_model_size, im_size(1)+1);
end
%sigma_tex = mean(D_tex_mse);
% ML sigma smooth estimate
sigma_smooth = sigma_smooth / depth_weight_count;
% ML sigma prior estimate
sigma_prior = mean(D_prior_mse);

end

