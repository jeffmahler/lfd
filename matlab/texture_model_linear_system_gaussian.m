function [gram_matrices, target_vectors, filter_banks, varargout] = ...
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
% and the max-likelihood depth prior (vectorized) is
%   D_vec_prior

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
    %duration = toc(start_time);
    %if mod(k-1, config.out_rate) == 0 && config.debug
       %fprintf('Loading image pyramid took %.03f sec\n', duration);
    %end
    %start_time = tic;
    [Phi, ~] = ...
        extract_texture_features(image_pyr, filter_banks, config);
    duration = toc(start_time);
    
    if mod(k-1, config.out_rate) == 0 && config.debug
       fprintf('Feature extraction took %.03f sec\n', duration);
    end
    num_pix = image_pyr.im_height * image_pyr.im_width;
    num_features = size(Phi,2);
    
    % save feature rep if desired
    if config.save_individual_features
        feature_filename = sprintf(config.feature_file_template, img_num);
        
        start_time = tic;
        save(feature_filename, 'Phi');
        duration = toc(start_time);
        
        if mod(k-1, config.out_rate) == 0 && config.debug
            fprintf('Saving took %.03f sec\n', duration);
        end
    end
    
    % compute normals
    %disp('Before compute norms');
    N_im = compute_normals(image_pyr, config);
    N_vec = reshape(N_im, [num_pix, 3]);
    %disp('After compute norms');
    
    % form target depth_vector
    D_vec = double(image_pyr.D_pyr{1});
    D_vec = D_vec(:);
    D_target_vec = D_vec;
    if config.use_inv_depth
       D_target_vec = config.max_depth ./ (D_vec + 1);
    end
    if config.use_log_depth
       D_target_vec = log(D_vec + 1);
    end
    
    % compute Phi indices for local models
    lin_ind = 1:num_pix;
    [row_ind, ~] = ind2sub([image_pyr.im_height, image_pyr.im_width], lin_ind);
    
    % rank-1 update to the linear system
    if exist('gram_matrices', 'var') && exist('target_vectors', 'var')
%         gram_matrix = gram_matrix + Phi' * Phi;
%         target_vector = target_vector + Phi' * D_target_vec;
    else
        vert_model_size = ceil(image_pyr.im_height / config.num_vert_models);
        gram_matrices = cell(1,vert_model_size);
        target_vectors = cell(1,vert_model_size);
        normal_vectors = cell(3,vert_model_size);
        init_matrices = 1;
%         gram_matrix = Phi' * Phi;
%         target_vector = Phi' * D_target_vec; 
    end
    
    % accumulate vertical models
    %disp('Before mat mult');
    start_row = 1;
    end_row = min(start_row + vert_model_size, image_pyr.im_height+1);
    for j = 1:config.num_vert_models
        v_row_ind = row_ind >= start_row & row_ind < end_row;
        v_lin_ind = lin_ind(v_row_ind);
        Phi_v = Phi(v_lin_ind,:);
        D_v = D_target_vec(v_lin_ind);
        
        if init_matrices
            gram_matrices{j} = Phi_v' * Phi_v;
            target_vectors{j} = Phi_v' * D_v;
            for n = 1:3
                N_v = N_vec(v_lin_ind,n);
                normal_vectors{n, j} = Phi_v' * N_v;
            end
        else
            gram_matrices{j} = gram_matrices{j} + Phi_v' * Phi_v;
            target_vectors{j} = target_vectors{j} + Phi_v' * D_v;
            for n = 1:3
                N_v = N_vec(v_lin_ind,n);
                normal_vectors{n, j} = normal_vectors{n, j} + Phi_v' * N_v;
            end
        end
        
        start_row = start_row + vert_model_size;
        end_row = min(start_row + vert_model_size, image_pyr.im_height+1);
    end
    if init_matrices
        init_matrices = 0;
    end
    %disp('After mat mult');

    % accumulate simple depth prior avg
    if nargout > 4
        if ~exist('D_prior_vec', 'var')
            D_prior_vec = zeros(num_pix, 1);
        end
        D_prior_vec = D_prior_vec + D_target_vec;
    end
    
    % save stacked phis
    if nargout > 5
        if ~exist('Phi_stacked', 'var')
            Phi_stacked = zeros(num_training*num_pix, num_features);
            start_I_p = 1;
            end_I_p = start_I_p + num_pix - 1;
        end
        Phi_stacked(start_I_p:end_I_p,:) = Phi;
        start_I_p = start_I_p + num_pix;
        end_I_p = start_I_p + num_pix-1;
    end
    
    % save stacked target_depths
    if nargout > 6
        if ~exist('D_target_stacked', 'var')
            D_target_stacked = zeros(num_training*num_pix, 1);
            start_I_d = 1;
            end_I_d = start_I_d + num_pix - 1;
        end
        D_target_stacked(start_I_d:end_I_d,:) = D_target_vec;
        start_I_d = start_I_d + num_pix;
        end_I_d = start_I_d + num_pix - 1;
    end
        
    % for debugging only
    if sum(isnan(target_vectors{1})) > 0
       stop = 1; 
    end
    
    % snapshot to prevent training data loss
    %disp('Before save');
    if mod(k-1, config.snapshot_rate) == 0 || k == (num_training - 1)
        training_snapshot.gram_matrix = gram_matrices;
        training_snapshot.target_vector = target_vectors;
        training_snapshot.normal_vectors = normal_vectors;
        training_snapshot.iter = k;
        training_snapshot.rgb_filename = rgb_filename;

        save(config.tmp_training_file, 'training_snapshot');
    end
    %disp('After save');
    
    if sum(sum(isnan(normal_vectors{1,1}))) > 0
        stop = 1;
    end
end

if nargout > 3
    varargout{1} = normal_vectors;
end
if nargout > 4
    varargout{2} = D_prior_vec / num_training;
end
if nargout > 5
    varargout{3} = Phi_stacked;
end
if nargout > 6
    varargout{4} = D_target_stacked;
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
