function [D_nom_error, D_sq_error, D_pred, corrs] = ...
    depth_error_gaussian_tps(matches, model, config, corrs)
%PREDICT_DEPTHS_GAUSSIAN _TPS

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

test_image_nums = matches(1,:);
train_image_nums = matches(2,:);
num_pred = size(test_image_nums, 2);
num_train = size(train_image_nums, 2);
if num_pred ~= num_train
   fprintf('Warning: Different number of train / test.'); 
end

new_corrs = 0;
if nargin < 4
    corrs = cell(1, num_pred);
    new_corrs = 1;
end

D_pred = cell(1, num_pred);
hund_um_to_m = config.hund_um_to_m;

% loop through training set and predict each with a linear function 
for k = 1:num_pred
    % read the next filename
    train_img_num = train_image_nums(k);
    test_img_num = test_image_nums(k);
    
    % form filenames
    train_rgb_filename = sprintf(config.rgb_file_template, train_img_num);
    train_depth_filename = sprintf(config.depth_file_template, train_img_num);
    
    test_rgb_filename = sprintf(config.rgb_file_template, test_img_num);
    test_depth_filename = sprintf(config.depth_file_template, test_img_num);
    
    if mod(k-1, config.out_rate) == 0
        fprintf('Computing tex error for image %d: %s\n', k, test_rgb_filename);
    end
    train_image_pyr = load_image_pyramid(train_rgb_filename, train_depth_filename, config);
    
    % load an RGBD pair and extract features
    test_image_pyr = load_image_pyramid(test_rgb_filename, test_depth_filename, config);
    [Phi, ~] = ...
        extract_texture_features(test_image_pyr, filter_banks, config); 
    [smooth_weights, ~] = ...
         create_texture_diff_weights(test_image_pyr, Phi, num_filters, config);
   
    if ~exist('D_nom_error', 'var') || ~exist('D_sq_error', 'var')
        num_pix = test_image_pyr.im_height * test_image_pyr.im_width;
        D_nom_error = zeros(num_pix, num_pred);
        D_sq_error = zeros(num_pix, num_pred);
    end
    
    % extract correspondences
    if new_corrs
        [train_corrs, test_corrs] = ...
            get_corrs(train_image_pyr.I_pyr{1}, test_image_pyr.I_pyr{1}, ...
            config.corr_scale);
        corrs{k} = round([test_corrs, train_corrs]);
    else
        if config.use_dense_corrs
            % compute dense correspondences
            train_I_rgb = read_and_downsample(train_rgb_filename, config.start_level);
            test_I_rgb = read_and_downsample(test_rgb_filename, config.start_level);

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

            test_corrs = [X(:), Y(:)];
            train_corrs = [corr_x(:), corr_y(:)];
            corrs{k} = round([test_corrs, train_corrs]);
        else
            cur_corrs = corrs{k};
            clean_corrs = fix_corrs(cur_corrs, train_image_pyr.im_height, ...
                train_image_pyr.im_width, config.start_level);
            train_corrs = clean_corrs(:,3:4);
            test_corrs = clean_corrs(:,1:2);
        end
    end
    num_corrs = size(train_corrs, 1);
    
    % predict depth based on texture model (w/o TPS)
    D_pred_vec = predict_depth_gaussian(test_image_pyr, Phi, smooth_weights, ...
        model, config);
        
    % invert depth back to true scale
    if config.use_inv_depth
       D_pred_vec = (config.max_depth ./ D_pred_vec) - 1;
    end
    % exponentiate to get depth
    if config.use_log_depth
       D_pred_vec = exp(D_pred_vec) - 1;
    end
    
    % get indices of corrs and depth at correspondences
    train_size = size(train_image_pyr.D_pyr{1});
    train_lin_ind = sub2ind(train_size, round(train_corrs(:,2)), round(train_corrs(:,1)));
    D_tps_train_corrs_vec = hund_um_to_m * double(train_image_pyr.D_pyr{1}(train_lin_ind));
    
    test_size = size(test_image_pyr.D_pyr{1});
    test_lin_ind = sub2ind(test_size, round(test_corrs(:,2)), round(test_corrs(:,1)));
    
    % start test estimate with training depths
    tps_corr_weights = ones(num_corrs, 1);
    tps_weights = ones(num_pix, 1);
    D_pred_vec(test_lin_ind)= D_tps_train_corrs_vec / hund_um_to_m;
    
    % get test normalized image coords
    D_nc_test = ones(test_image_pyr.im_height, test_image_pyr.im_width);
    test_pts_nc = project_depth_im(D_nc_test, test_image_pyr.K);
    corr_ind = [test_lin_ind, train_lin_ind];
    
    % form target depth_vector
    D_vec = double(test_image_pyr.D_pyr{1});
    D_target_vec = D_vec(:);
    
    % alternate tps and depth prediction
    for i = 1:config.num_tps_iter
 %       config.bend_coef = 2.0 * config.bend_coef;
        % convert latest tps prediction
        D_tps_test_corrs_vec = hund_um_to_m * double(D_pred_vec(test_lin_ind));
        %D_tps_test_vec = D_tps_train_vec;
        
        % fit thin plate spline (subtract 1 because of matlab indexing)
        fprintf('Fitting tps iter %d\n', i);
        [tps, ~, ~, ~] = tps_fit_depth_im(D_tps_train_corrs_vec, D_tps_test_corrs_vec, ...
            train_corrs-1, test_corrs-1, train_image_pyr.K, test_image_pyr.K, ...
            tps_corr_weights, config);
        %test_3d_pred = tps_apply(tps, train_3d); % for debugging only
        
        % project ALL training points into 3d
        D_im_train = hund_um_to_m * double(train_image_pyr.D_pyr{1});
        D_im_test = hund_um_to_m * double(test_image_pyr.D_pyr{1});
        train_pts_3d = project_depth_im(D_im_train, train_image_pyr.K);
        test_pts_3d = project_depth_im(D_im_test, test_image_pyr.K);
        test_pts_3d_pred = tps_apply(tps, train_pts_3d); % will not work if train / test are different sizes
        
        % for visualization only
        [D_im_test, tps_weights] = project_depth_pts(test_pts_3d_pred, test_image_pyr.im_height, ...
            test_image_pyr.im_width, test_image_pyr.K);
        figure(17);
        subplot(1,3,1);
        imshow(histeq(train_image_pyr.D_pyr{1}));
        title('True Train');
        subplot(1,3,2);
        imshow(histeq(test_image_pyr.D_pyr{1}));
        title('True Test');
        subplot(1,3,3);
        imshow(histeq(uint16(D_im_test / hund_um_to_m)));
        title('TPS Warped');
%         scatter3(test_pts_3d_pred(:,1), test_pts_3d_pred(:,2), test_pts_3d_pred(:,3));
%         xlabel('X');
%         ylabel('Y');
%         zlabel('Z');
%         
        % convert back to uint16, remove negative depths
%        D_tps_pred_vec = D_im_test(:) / hund_um_to_m; 
%        D_tps_pred_vec(D_tps_pred_vec <= 0) = 0;
        
%         figure(18);
%         D_tps_im = reshape(D_tps_pred_vec, 60, 80);
%         imshow(histeq(uint16(D_tps_im)));
        
        % add tps 'regularizer' to texture-based predictor
%         model.D_tps_vec = D_tps_pred_vec;
%         if config.use_inv_depth
%            model.D_tps_vec = config.max_depth ./ (D_tps_pred_vec + 1);
%         end
%         % log depth transform
%         if config.use_log_depth
%            model.D_tps_vec = log(D_tps_pred_vec + 1);
%         end
        
        % predict depth based on texture model (w TPS)
        fprintf('Predicting depth iter %d\n', i);
        %D_pred_vec = model.D_tps_vec;
        D_pred_vec = predict_depth_gaussian_tps(test_image_pyr, Phi, ...
            corr_ind, test_pts_nc, test_pts_3d_pred, ...
            smooth_weights, tps_weights, model, config);
        D_t_pred_vec = predict_depth_gaussian(test_image_pyr, Phi, smooth_weights, ...
            model, config);
        
        % invert depth back to true scale
        if config.use_inv_depth
           D_pred_vec = (config.max_depth ./ D_pred_vec) - 1;
        end
        % exponentiate to get depth
        if config.use_log_depth
           D_pred_vec = exp(D_pred_vec) - 1;
           D_t_pred_vec = exp(D_t_pred_vec) - 1;
        end
        
        D_t_im = reshape(D_t_pred_vec, test_image_pyr.im_height, test_image_pyr.im_width);
        D_pred{k} = reshape(D_pred_vec, test_image_pyr.im_height, test_image_pyr.im_width);
        
        % visualization only
        figure(18);
        subplot(1,3,1);
        imshow(histeq(test_image_pyr.D_pyr{1}));
        title('True Test');
        subplot(1,3,2);
        imshow(histeq(uint16(D_pred{k}))); 
        title('Predicted Test');
        subplot(1,3,3);
        imshow(histeq(uint16(D_t_im))); 
        title('Predicted Test (w/o TPS)');
    end
    
    % image for visualization
    D_pred{k} = reshape(D_pred_vec, test_image_pyr.im_height, test_image_pyr.im_width);
        
    if config.vis_pred
        figure(11);
        subplot(1,2,1);
        imshow(histeq(test_image_pyr.D_pyr{1}));
        title('Ground Truth');
        subplot(1,2,2);
        imshow(histeq(uint16(D_pred{k}))); 
        title('Predicted');
    end
%    D_tps_error = hund_um_to_m * tps_weights .* (D_pred_vec - D_target_vec);
    
    D_nom_error(:,k) = hund_um_to_m * (D_pred_vec - D_target_vec);
    D_sq_error(:,k) = (D_nom_error(:,k)).^2;
    %D_mse(k) = mean(D_sq_error);
end

if config.vis_pred_hist
    figure(10);
    high = prctile(D_nom_error(:), 100 - config.hist_prctile);
    low = prctile(D_nom_error(:), config.hist_prctile);
    bdry = max(abs(high), abs(low));
    bin_width = 2*bdry / config.n_bins;
    bin_edges = -bdry:bin_width:bdry;
    h = histc(D_nom_error(:), bin_edges);
    bar(bin_edges, h, 'histc');
    %hist(D_nom_error(:), config.n_bins);
    title('Depth Error');
    xlim([-bdry, bdry]);
end

if config.disp_pred
    fprintf('Nominal Error\n');
    fprintf('Mean:\t%.03f\n', mean(D_nom_error(:)));
    fprintf('Med:\t%.03f\n', median(D_nom_error(:)));
    fprintf('Std:\t%.03f\n', std(D_nom_error(:)));

    fprintf('\nSquared Error\n');
    fprintf('Mean:\t%.03f\n', mean(D_sq_error(:)));
    fprintf('Med:\t%.03f\n', median(D_sq_error(:)));
    fprintf('Min:\t%.03f\n', min(D_sq_error(:)));
    fprintf('Max:\t%.03f\n', max(D_sq_error(:)));
end

end

function I_small = read_and_downsample(filename, level)
    I = imread(filename);
    I_small = I;
    for i = 1:(level-1)
        I_small = impyramid(I_small, 'reduce');
    end
end


