% script to train a gaussian texture model
config = struct();
rng(100);

% training set
config.data_dir = 'data/VOCB3DO';
config.out_dir = 'results/linear_gaussian';
config.rgb_dir = 'KinectColor';
config.depth_dir = 'RegisteredDepthData';
config.feature_dir = 'Features';
config.corr_dir = 'Correspondences';
config.rgb_name = 'img_%04d.png';
config.depth_name = 'img_%04d_abs_smooth.png';
config.feature_name = 'img_%04d_features.mat';
config.training_pct = 0.94;
config.validation_pct = 0.03;
config.total_num_files = 849;  % number of files in kinectdata 
config.training_nums = randsample(config.total_num_files, ...
   uint16(config.training_pct * config.total_num_files)) - 1;

% get test / validation
test_valid_nums = setdiff(0:(config.total_num_files-1), config.training_nums)';
num_left = size(test_valid_nums, 1);
validation_ind = randsample(num_left, ...
   uint16(config.validation_pct * config.total_num_files));
config.validation_nums = test_valid_nums(validation_ind);
config.test_nums = setdiff(test_valid_nums, config.validation_nums);

% config.training_nums = [4; 5; 8; 1; 0; 7];
% config.test_nums = [3; 6; 9; 2];

% data reading
config.num_levels = 3;
config.start_level = 4;
config.level_rate = 1;
config.color_space = 'ycbcr';
config.rgb_file_template = ...
    sprintf('%s/%s/%s', config.data_dir, config.rgb_dir, config.rgb_name);
config.depth_file_template = ...
    sprintf('%s/%s/%s', config.data_dir, config.depth_dir, config.depth_name);
config.feature_file_template = ...
    sprintf('%s/%s/%s', config.data_dir, config.feature_dir, config.feature_name);

% features
config.use_texture_energy = 1;
config.use_energy = 1;
config.use_kurtosis = 1;
config.patch_size_x = 5;
config.patch_size_y = 5;
config.save_individual_features = 0;
config.extract_features = 1;
config.corr_scale = 4.0;
config.tex_bandwidth = 2.5;

% filters
config.use_serge_filts = 1;
config.use_doog_filts = 0;
config.use_LM_filts = 0;
config.use_LW_filts = 1;
config.filt_size = 13.0;

% depth params
config.use_inv_depth = 0;
config.use_log_depth = 0;
config.grad_scale = 5e-3; % higher means more gradients
config.max_depth = 36155.0;
config.focal = 531.0;

% learning params
config.num_vert_models = 10;
config.beta = 1e2;
config.tol = 1e-6;
config.max_iters = 300;

% corrs
config.label_corrs = 1;
config.num_matches = 5;

% tps
config.bend_coef = 0.1;
config.rot_coef = 10.0;

% prediction
config.n_bins = 100;
config.hist_prctile = 5;
config.vis_pred = 1;
config.vis_pred_hist = 1;
config.disp_pred = 1;

% snapshots / debug
config.show = 0;
config.vis = 0;
config.debug = 1;
config.tmp_training_file = sprintf('%s/tmp_training_system.mat', config.out_dir);
config.out_rate = 1;
config.snapshot_rate = 50;
config.hund_um_to_m = 1e-4;

%% create gist db
gist_db = create_gist_database(config);
save(sprintf('%s/gist_db', config.out_dir), 'gist_db');

%% manually label corrs (if specified)
corr_dir = sprintf('%s/%s', config.data_dir, config.corr_dir);
valid_corr_filename = sprintf('%s/validation_corrs.mat', corr_dir);
test_corr_filename = sprintf('%s/test_corrs.mat', corr_dir);
    
if config.label_corrs % TODO: change to check for corrs in filesystem first
    validation_corrs = struct();
    test_corrs = struct();
    
    % label the correspondences between the images
    [validation_corrs.matches, validation_corrs.corrs] = ...
        label_corrs(config.validation_nums, gist_db, config);
    [test_corrs.matches, test_corrs.corrs] = ...
        label_corrs(config.test_nums, gist_db, config);
    
    % save the correspondences for future use
    save(valid_corr_filename, 'validation_corrs');
    save(test_corr_filename, 'test_corrs');
else
    load(valid_corr_filename);
    load(test_corr_filename);
end

%% extract features
linear_system = struct();
[linear_system.A, linear_system.b, filter_banks, linear_system.n_b, D_prior] = ...%, Phi_stacked, D_target_stacked] =  ...
    texture_model_linear_system_gaussian(config);
save(sprintf('%s/linear_system', config.out_dir), 'linear_system');
save(sprintf('%s/config', config.out_dir), 'config');

% save(sprintf('%s/Phi_stacked.mat', config.out_dir), 'Phi_stacked');
% save(sprintf('%s/D_target_stacked.mat', config.out_dir), 'D_target_stacked');

%% learn a model
disp('Learning linear model');
model = struct();
num_features = size(linear_system.A{1},1);

% gaussian model
model.num_vert_predictors = config.num_vert_models;
model.w = cell(1, model.num_vert_predictors);
model.theta = cell(3, model.num_vert_predictors); 
for i = 1:model.num_vert_predictors
    fprintf('Learning weights %d\n', i);
    model.w{i} = (linear_system.A{i} + config.beta * eye(num_features)) \ linear_system.b{i};
    for j = 1:3
        model.theta{j,i} = (linear_system.A{i} + config.beta * eye(num_features)) \ linear_system.n_b{j, i};
    end
end
% lasso model
% [B, fit_info] = ...
%     lasso(Phi_stacked(:,1:(num_features-1)), D_target_stacked, 'Lambda', config.beta);
% %lassoPlot(B,fit_info,'PlotType','CV');
% model.w = [B(:,1); fit_info.Intercept];

model.beta = config.beta;
model.D_prior_vec = D_prior;

%% learn max likelihood variances
%config.grad_scale = 1e-4;
% [model.sigma_tex, model.sigma_smooth, model.sigma_prior] = ...
%     learn_sigma_tex(model, config);
% save(sprintf('%s/model.mat', config.out_dir), 'model');

config.cellsize = 3;
config.gridspacing = 1;
sift_flow_params.alpha=2*255;
sift_flow_params.d=40*255;
sift_flow_params.gamma=0.005*255;
sift_flow_params.nlevels=3;
sift_flow_params.wsize.x=5;
sift_flow_params.wsize.y=5;
sift_flow_params.topwsize.x=10;
sift_flow_params.topwsize.y=3;
sift_flow_params.nTopIterations = 60;
sift_flow_params.nIterations= 30;
config.sift_flow_params = sift_flow_params;
config.use_dense_corrs = 1;

%config.num_tps_iter = 10;
config.bend_coef = 1.0;
config.rot_coef = 1e-4;
[model.sigma_tex, model.sigma_smooth, model.sigma_prior, model.sigma_tps] = ...
    learn_sigma_validation(model, validation_corrs.matches, ...
        validation_corrs.corrs, config);
save(sprintf('%s/model.mat', config.out_dir), 'model');

%% training error
disp('Computing training error');
train_error = struct();
[train_error.D_nom_error, train_error.D_sq_error, train_D_pred] = ...
    depth_error_gaussian(config.training_nums, model, config);
save(sprintf('%s/training_error.mat', config.out_dir), 'train_error');

%% validation error
disp('Computing validation error');
config.num_tps_iter = 10;
validation_error = struct();
[validation_error.D_nom_error, validation_error.D_sq_error, validation_D_pred] = ...
    depth_error_gaussian_tps(validation_corrs.matches, model, config, validation_corrs.corrs);
save(sprintf('%s/validation_error.mat', config.out_dir), 'validation_error');

%% test error
disp('Computing test error');
test_error = struct();
[test_error.D_nom_error, test_error.D_sq_error, test_D_pred] = ...
    depth_error_gaussian(config.test_nums, model, config);
save(sprintf('%s/test_error.mat', config.out_dir), 'test_error');

%% test error (tps version)
% disp('Computing test error');
% 
% load('data/VOCB3DO/Features/siftflow_corrs.mat');
% model.sigma_tps = 10*model.sigma_tex; % makes for weights = 1 later
% %model.sigma_smooth = 0.01*model.sigma_tex; % makes for weights = 1 later
% %model.sigma_prior = 1*model.sigma_tex; % makes for weights = 1 later
% config.num_tps_iter = 10;
% config.bend_coef = 10;
% config.rot_coef = 1e-4;
% 
% test_error = struct();
% [test_error.D_nom_error, test_error.D_sq_error, test_D_pred, corrs] = ...
%     depth_error_gaussian_tps(config.training_nums(207), ...
%     config.test_nums(14), model, config, corrs);
% %save(sprintf('%s/test_error.mat', config.out_dir), 'test_error');

%% visualize prior
% figure(3);
% D_im_prior = reshape(model.D_prior_vec, 60, 80);
% D_im_prior = exp(D_im_prior) - 1;
% imshow(histeq(uint16(D_im_prior)));
% title('Depth Prior');

%% redo hist
% D_nom_error = train_error.D_nom_error;
% figure(11);
% high = prctile(D_nom_error(:), 100 - config.hist_prctile);
% low = prctile(D_nom_error(:), config.hist_prctile);
% bdry = max(abs(high), abs(low));
% bin_width = 2*bdry / config.n_bins;
% bin_edges = -bdry:bin_width:bdry;
% h = histc(D_nom_error(:), bin_edges);
% bar(bin_edges, h, 'histc');
% %hist(D_nom_error(:), config.n_bins);
% title('Depth Error');
% xlim([-bdry, bdry]);

%%
% old_linear_system = linear_system;
% old_model = model;
% 
% for i = 2:config.num_vert_models
%     linear_system.A{1} = linear_system.A{1} + linear_system.A{i};
%     linear_system.b{1} = linear_system.b{1} + linear_system.b{i};
% end
% config.num_vert_models = 1;

%%
% Nominal Error
% Mean:	0.430
% Med:	0.090
% Std:	1.042
% 
% Squared Error
% Mean:	1.269
% Med:	0.137
% Min:	0.000
% Max:	50.178

