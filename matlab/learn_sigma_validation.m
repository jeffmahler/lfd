function [sigma_tex, sigma_smooth, sigma_prior, sigma_tps] = ...
    learn_sigma_validation(model, matches, corrs, config)
%LEARN_SIGMA_VALIDATION learn sigma values on a validation set of images

% create filter banks
cur_index = 1;
d = 2; % dimension of filters
num_possible_filts = 5;
filter_banks = cell(1,num_possible_filts);
num_filters = 0;
hund_um_to_m = config.hund_um_to_m;

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

num_validation = size(config.validation_nums, 1);
%D_tex_mse = zeros(num_training, 1);
D_prior_mse = zeros(num_validation, 1);
D_tps_mse = zeros(num_validation, 1);

depth_weight_count = 0.0;
sigma_smooth = 0.0;

% loop through training set and predict each with a linear function 
for k = 1:num_validation
    % read the next filename
    img_num = config.validation_nums(k);
    rgb_filename = sprintf(config.rgb_file_template, img_num);
    depth_filename = sprintf(config.depth_file_template, img_num);
    cur_match = matches(:,k);
    cur_corrs = corrs{k};
    
    if mod(k-1, config.out_rate) == 0
        fprintf('Computing tex error for image %d: %s\n', k, rgb_filename);
    end
    
    % load an RGBD pair
    test_image_pyr = load_image_pyramid(rgb_filename, depth_filename, config);
    [Phi, ~] = ...
        extract_texture_features(test_image_pyr, filter_banks, config);
    [~, D_diff] = ...
         create_texture_diff_weights(test_image_pyr, Phi, num_filters, config);
    im_size = [test_image_pyr.im_height, test_image_pyr.im_width];
    num_pix = im_size(1) * im_size(2);
    
    % get normals
    N_im = compute_normals(test_image_pyr, config);
    N_target_vec = reshape(N_im, [num_pix, 3]);
     
    % form target depth_vector
    D_vec = double(test_image_pyr.D_pyr{1});
    D_vec = D_vec(:);
    D_target_vec = D_vec;
    if config.use_inv_depth
       D_target_vec = config.max_depth ./ (D_vec + 1);
    end
    if config.use_log_depth
       D_target_vec = log(D_vec+1);
    end
 
    % update max-likelihood tex depth error
    lin_ind = 1:num_pix;
    [row_ind, ~] = ind2sub(im_size, lin_ind);
    D_pred_vec = zeros(num_pix, 1);
    N_pred_vec = zeros(num_pix, 3);
    vert_model_size = ceil(im_size(1) / config.num_vert_models);
        
    % predict depth per-row
    start_row = 1;
    end_row = min(start_row + vert_model_size, im_size(1)+1);
    for j = 1:config.num_vert_models
        v_row_ind = row_ind >= start_row & row_ind < end_row;
        v_lin_ind = lin_ind(v_row_ind);
        Phi_v = Phi(v_lin_ind,:);
        
        D_pred_vec(v_lin_ind) = Phi_v * model.w{j};
        for a = 1:3
            N_pred_vec(v_lin_ind, a) = Phi_v * model.theta{a, j};
        end
        
        start_row = start_row + vert_model_size;
        end_row = min(start_row + vert_model_size, im_size(1)+1);
    end
    
    % normalize predictions
    N_pred_mag = sqrt(sum(N_pred_vec.^2, 2));
    N_pred_mag_mat = repmat(N_pred_mag, 1, 3);
    N_pred_vec = N_pred_vec ./ N_pred_mag_mat;
    
    % load corresponding training image
    assert(cur_match(1) == img_num);
    train_img_num = cur_match(2);
    
    train_rgb_filename = sprintf(config.rgb_file_template, train_img_num);
    train_depth_filename = sprintf(config.depth_file_template, train_img_num);
    train_image_pyr = load_image_pyramid(train_rgb_filename, train_depth_filename, config);
    
    if config.use_dense_corrs
        % compute dense correspondences
        train_I_rgb = read_and_downsample(train_rgb_filename, config.start_level);
        test_I_rgb = read_and_downsample(rgb_filename, config.start_level);
        
        sift_train = mexDenseSIFT(train_I_rgb, config.cellsize, config.gridspacing);
        sift_test = mexDenseSIFT(test_I_rgb, config.cellsize, config.gridspacing);
    
        [vx, vy, ~] = SIFTflowc2f(sift_test, sift_train, config.sift_flow_params);
        
        [X, Y] = meshgrid(1:test_image_pyr.im_width, 1:test_image_pyr.im_height);
        
        corr_x = X + vx;
        corr_y = Y + vy;
        
        corr_x(corr_x < 1) = 1;
        corr_y(corr_y < 1) = 1;
        corr_x(corr_x > train_image_pyr.im_width) = train_image_pyr.im_width;
        corr_y(corr_y > train_image_pyr.im_height) = train_image_pyr.im_height;
        
        test_pts = [X(:), Y(:)];
        train_pts = [corr_x(:), corr_y(:)];
    else
        % load in corrs
        clean_corrs = fix_corrs(cur_corrs, train_image_pyr.im_height, ...
            train_image_pyr.im_width, config.start_level);
        train_pts = clean_corrs(:,3:4);
        test_pts = clean_corrs(:,1:2);
    end

    num_corrs = size(test_pts,1);
    tps_corr_weights = ones(num_corrs, 1);
    
    % get depths at corresponding points
    train_size = size(train_image_pyr.D_pyr{1});
    train_lin_ind = sub2ind(train_size, round(train_pts(:,2)), round(train_pts(:,1)));
    D_tps_train_vec = hund_um_to_m * double(train_image_pyr.D_pyr{1}(train_lin_ind));
    
    test_size = size(test_image_pyr.D_pyr{1});
    test_lin_ind = sub2ind(test_size, round(test_pts(:,2)), round(test_pts(:,1)));
    D_tps_test_vec = hund_um_to_m * double(test_image_pyr.D_pyr{1}(test_lin_ind));
    
    % predict depth using tps only
    [tps, ~, ~, ~] = tps_fit_depth_im(D_tps_train_vec, D_tps_test_vec, ...
        train_pts-1, test_pts-1, train_image_pyr.K, test_image_pyr.K, ...
        tps_corr_weights, config);
        
    % project ALL training points into 3d
    D_im_train = hund_um_to_m * double(train_image_pyr.D_pyr{1});
    train_pts_3d = project_depth_im(D_im_train, train_image_pyr.K);
    D_im_test = hund_um_to_m * double(test_image_pyr.D_pyr{1});
    test_pts_3d = project_depth_im(D_im_test, test_image_pyr.K);
    test_pts_3d_pred = tps_apply(tps, train_pts_3d);
    
    test_pts_3d_corr = test_pts_3d(test_lin_ind, :);
    test_pts_3d_pred_corr = test_pts_3d_pred(train_lin_ind, :);
%     [D_tps_pred_im, ~] = project_depth_pts(test_pts_3d_pred, test_image_pyr.im_height, ...
%         test_image_pyr.im_width, test_image_pyr.K);
%     D_tps_pred_vec = D_tps_pred_im(:) / hund_um_to_m; % transform?
    
%     D_im = reshape(exp(D_pred_vec)-1, im_size);
%     figure(13);
%     imshow(histeq(uint16(D_im)));
%     hold on;
%     for j = 1:config.num_vert_models
%         plot([0;im_size(2)], [vert_model_size*(j-1); vert_model_size*(j-1)]);
%     end
%     pause(10);
    
    % update ML texture sigma
    D_sq_error = (D_pred_vec - D_target_vec).^2;
    if ~exist('D_tex_mse', 'var')
        D_tex_mse = zeros(num_pix, 1);
    end
    D_tex_mse = D_tex_mse + D_sq_error;
    
    % update max-likelihood smoothness param
    height = test_image_pyr.im_height;
    width = test_image_pyr.im_width;
    num_pix = height*width;
    sigma_smooth = sigma_smooth + sum(sum(D_diff{1}));
    sigma_smooth = sigma_smooth + sum(sum(D_diff{2}));
    sigma_smooth = sigma_smooth + sum(sum(D_diff{3}));
    sigma_smooth = sigma_smooth + sum(sum(D_diff{4}));
    depth_weight_count = depth_weight_count + 4 * num_pix;
    
    % update max-likelihood prior
    D_prior_sq_error = (model.D_prior_vec - D_target_vec).^2;
    D_prior_mse(k) = mean(D_prior_sq_error);
    
    % update tps ML sigma
    D_tps_sq_error = (test_pts_3d_corr - test_pts_3d_pred_corr).^2; %don't use transformed version
    D_tps_mse(k) = mean(sum(D_tps_sq_error, 2));
    
    if k == 21
        stop = 1;
    end
    
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
        (num_validation * size(v_lin_ind,2));
    
    start_row = start_row + vert_model_size;
    end_row = min(start_row + vert_model_size, im_size(1)+1);
end
%sigma_tex = mean(D_tex_mse);

% ML sigma smooth estimate
sigma_smooth = sigma_smooth / depth_weight_count;

% ML sigma prior estimate
sigma_prior = mean(D_prior_mse);

%ML sigma tps estimate
sigma_tps = mean(D_tps_mse);
end

function I_small = read_and_downsample(filename, level)
    I = imread(filename);
    I_small = I;
    for i = 1:(level-1)
        I_small = impyramid(I_small, 'reduce');
    end
end

