function [D_nom_error, D_sq_error, D_pred] = ...
    depth_error_gaussian(image_nums, model, config)
%PREDICT_DEPTHS_GAUSSIAN 

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

num_pred = size(image_nums, 1);
D_pred = cell(1, num_pred);

% loop through training set and predict each with a linear function 
for k = 1:num_pred
    % read the next filename
    img_num = image_nums(k);
    rgb_filename = sprintf(config.rgb_file_template, img_num);
    depth_filename = sprintf(config.depth_file_template, img_num);
    
    if mod(k-1, config.out_rate) == 0
        fprintf('Computing tex error for image %d: %s\n', k, rgb_filename);
    end
    
    % load an RGBD pair and extract features
    image_pyr = load_image_pyramid(rgb_filename, depth_filename, config);
    [Phi, I_gradients] = ...
        extract_texture_features(image_pyr, filter_banks, config); 
    [weights, ~] = ...
         create_texture_diff_weights(image_pyr, Phi, num_filters, config);
    % [weights, ~] = ...
    %    create_depth_diff_weights(image_pyr, I_gradients, config);
    
    if ~exist('D_nom_error', 'var') || ~exist('D_sq_error', 'var')
        num_pix = image_pyr.im_height * image_pyr.im_width;
        D_nom_error = zeros(num_pix, num_pred);
        D_sq_error = zeros(num_pix, num_pred);
    end
    
    % predict depth
    D_pred_vec = predict_depth_gaussian(image_pyr, Phi, weights, ...
        model, config);
    
    % invert depth back to true scale
    if config.use_inv_depth
       D_pred_vec = (config.max_depth ./ D_pred_vec) - 1;
    end
    % exponentiate to get depth
    if config.use_log_depth
       D_pred_vec = exp(D_pred_vec) - 1;
    end
    
    % form target depth_vector
    D_vec = double(image_pyr.D_pyr{1});
    D_target_vec = D_vec(:);
    
    D_pred{k} = reshape(D_pred_vec, image_pyr.im_height, image_pyr.im_width);
        
    if config.vis_pred
        figure(11);
        subplot(1,2,1);
        imshow(histeq(image_pyr.D_pyr{1}));
        title('Ground Truth');
        subplot(1,2,2);
        imshow(histeq(uint16(D_pred{k}))); 
        title('Predicted');
    end
 
    hund_um_to_m = 1e-4;
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

