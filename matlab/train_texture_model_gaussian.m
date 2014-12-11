% script to train a gaussian texture model
config = struct();
rng(100);

% training set
config.data_dir = 'data/VOCB3DO';
config.out_dir = 'results/log_linear_gaussian';
config.rgb_dir = 'KinectColor';
config.depth_dir = 'RegisteredDepthData';
config.feature_dir = 'Features';
config.rgb_name = 'img_%04d.png';
config.depth_name = 'img_%04d_abs_smooth.png';
config.feature_name = 'img_%04d_features.mat';
config.training_pct = 0.6;
config.total_num_files = 10;%849  % number of files in kinectdata 
config.training_nums = randsample(config.total_num_files, ...
   uint16(config.training_pct * config.total_num_files)) - 1;
config.test_nums = setdiff(0:(config.total_num_files-1), config.training_nums)';

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
config.patch_size_x = 3;
config.patch_size_y = 3;
config.save_individual_features = 0;
config.extract_features = 1;
config.corr_scale = 4.0;

% filters
config.use_serge_filts = 1;
config.use_doog_filts = 0;
config.use_LM_filts = 0;
config.use_LW_filts = 1;
config.filt_size = 13.0;

% depth params
config.use_inv_depth = 0;
config.use_log_depth = 1;
config.grad_scale = 5.0; % higher means more gradients
config.max_depth = 36155.0;
config.focal = 531.0;

% learning params
config.beta = 1e4;
config.tol = 1e-6;
config.max_iters = 300;

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
config.snapshot_rate = 10;
config.hund_um_to_m = 1e-4;

%% extract features
linear_system = struct();
[linear_system.A, linear_system.b, filter_banks, D_prior] = ...%, Phi_stacked, D_target_stacked] =  ...
    texture_model_linear_system_gaussian(config);
save(sprintf('%s/linear_system.mat', config.out_dir), 'linear_system');

%% learn a model
disp('Learning linear model');
model = struct();
num_features = size(linear_system.A,1);

% gaussian model
model.w = (linear_system.A + config.beta * eye(num_features)) \ linear_system.b;

% lasso model
% [B, fit_info] = ...
%     lasso(Phi_stacked(:,1:(num_features-1)), D_target_stacked, 'Lambda', config.beta);
% %lassoPlot(B,fit_info,'PlotType','CV');
% model.w = [B(:,1); fit_info.Intercept];

model.beta = config.beta;
model.D_prior_vec = D_prior;

%% learn max likelihood variances
[model.sigma_tex, model.sigma_smooth, model.sigma_prior] = ...
    learn_sigma_tex(model, config);
save(sprintf('%s/model.mat', config.out_dir), 'model');

%% training error
disp('Computing training error');
train_error = struct();
[train_error.D_nom_error, train_error.D_sq_error, train_D_pred] = ...
    depth_error_gaussian(config.training_nums, model, config);
save(sprintf('%s/training_error.mat', config.out_dir), 'train_error');

%% test error
disp('Computing test error');

%load('data/VOCB3DO/Features/corrs.mat');
model.sigma_tps = 0.2*model.sigma_tex; % makes for weights = 1 later
model.sigma_smooth = 0.1*model.sigma_tex; % makes for weights = 1 later
config.num_tps_iter = 30;
config.bend_coef = 1e-1;
config.rot_coef = 1e-4;
config.grad_scale = 1e-2;

test_error = struct();
%[test_error.D_nom_error, test_error.D_sq_error, test_D_pred] = ...
%    depth_error_gaussian(config.test_nums, model, config);
[test_error.D_nom_error, test_error.D_sq_error, test_D_pred, corrs] = ...
    depth_error_gaussian_tps(config.training_nums(3), ...
    config.test_nums(4), model, config, corrs);
save(sprintf('%s/test_error.mat', config.out_dir), 'test_error');

%% visualize prior
% figure(3);
% D_im_prior = reshape(model.D_prior_vec, 60, 80);
% D_im_prior = exp(D_im_prior) - 1;
% imshow(histeq(uint16(D_im_prior)));
% title('Depth Prior');

%% get corresponding points



