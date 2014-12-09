function [gram_matrix, target_vector, filter_banks] = ...
    texture_model_linear_system_gaussian(config)
% Computes a multiscale texture-based feature representation for each image
% in the training set specified by the config file
% To do this, it incrementally loads each RGBD pair, runs it through a
% filter bank, arranges features in a window-based fashion, creates depth
% difference weights for smoothing, and performs a rank-1 update to the
% relevant set of matrices
%
% Upon completion, the best linear model is
%   gram_matrix^{-1} * target_vector
% and the max-likelihood smoothing parameter is
%   sigma_smooth 

% matrices to store intermediate data
num_training = size(config.training_nums, 1);

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
training_snapshot = struct();

% loop through training set and 
for k = 1:num_training
    % read the next filename
    img_num = config.training_nums(k);
    rgb_filename = sprintf(config.rgb_file_template, img_num);
    depth_filename = sprintf(config.depth_file_template, img_num);
    
    if mod(k-1, config.out_rate) == 0
        fprintf('Processing image %d: %s\n', k, rgb_filename);
    end
    
    % extract relevant data for the current RGBD pair
    start_time = tic;
    image_pyr = load_image_pyramid(rgb_filename, depth_filename, config);
    [Phi, ~] = ...
        extract_texture_features(image_pyr, filter_banks, config);
    duration = toc(start_time);
    
    if mod(k-1, config.out_rate) == 0 && config.debug
       fprintf('Feature extraction took %.03f sec\n', duration);
    end
    
    % form target depth_vector
    D_vec = double(image_pyr.D_pyr{1});
    D_vec = D_vec(:);
    D_target_vec = D_vec;
    if config.use_inv_depth
       D_target_vec = config.max_depth ./ D_vec;
    end
    if config.use_log_depth
       D_target_vec = log(D_vec + 1);
    end
    
    % rank-1 update to the linear system
    if exist('gram_matrix', 'var') && exist('target_vector', 'var')
        gram_matrix = gram_matrix + Phi' * Phi;
        target_vector = target_vector + Phi' * D_target_vec;
    else
        num_features = size(Phi, 2);
        gram_matrix = Phi' * Phi;
        target_vector = Phi' * D_target_vec; 
    end
    
    if sum(isnan(target_vector)) > 0
       stop = 1; 
    end
    
    % snapshot to prevent training data loss
    if mod(k-1, config.snapshot_rate) == 0 || k == (num_training - 1)
        training_snapshot.gram_matrix = gram_matrix;
        training_snapshot.target_vector = target_vector;
        training_snapshot.iter = k;
        training_snapshot.rgb_filename = rgb_filename;

        save(config.tmp_training_file, 'training_snapshot');
    end
end

end

% Train
% Nominal Error
% Mean:	-0.145
% Med:	-0.009
% Std:	1.036
% 
% Squared Error
% Mean:	109489890.205
% Med:	10754583.033
% Min:	0.000
% Max:	96406535468965.266

% TEST
% Nominal Error
% Mean:	-0.166
% Med:	-0.022
% Std:	0.958
% 
% Squared Error
% Mean:	94572530.818
% Med:	10263174.169
% Min:	0.000
% Max:	46201553050344.328

% TEST grad mult = 0.1
% Nominal Error
% Mean:	-0.184
% Med:	-0.040
% Std:	0.946
% 
% Squared Error
% Mean:	92856818.611
% Med:	10307068.387
% Min:	0.000
% Max:	179801454081114.531
