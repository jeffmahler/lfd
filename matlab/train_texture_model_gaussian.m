% script to train a gaussian texture model
config = struct();
rng(100);

% training set
config.data_dir = 'data/VOCB3DO';
config.out_dir = 'results/log_linear_gaussian';
config.depth_dir = 'RegisteredDepthData';
config.rgb_dir = 'KinectColor';
config.rgb_name = 'img_%04d.png';
config.depth_name = 'img_%04d_abs_smooth.png';
config.training_pct = 0.6;
config.total_num_files = 100;%849  % number of files in kinectdata 
config.training_nums = randsample(config.total_num_files, ...
   uint16(config.training_pct * config.total_num_files)) - 1;
config.test_nums = setdiff(0:(config.total_num_files-1), config.training_nums)';
config.extract_features = 1;

% data reading
config.num_levels = 3;
config.start_level = 4;
config.level_rate = 1;
config.color_space = 'ycbcr';
config.rgb_file_template = ...
    sprintf('%s/%s/%s', config.data_dir, config.rgb_dir, config.rgb_name);
config.depth_file_template = ...
    sprintf('%s/%s/%s', config.data_dir, config.depth_dir, config.depth_name);

% filters
config.use_serge_filts = 1;
config.use_doog_filts = 0;
config.use_LM_filts = 0;
config.use_LW_filts = 1;
config.filt_size = 13.0;

% depth params
config.use_inv_depth = 0;
config.use_log_depth = 1;
config.grad_scale = 0.5; % higher means more gradients
config.max_depth = 36155.0;

% learning params
config.gamma = 1e4;
config.tol = 1e-6;
config.max_iters = 300;

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
config.out_rate = 10;
config.snapshot_rate = 10;

%% extract features
linear_system = struct();
[linear_system.A, linear_system.b, filter_banks] =  ...
    texture_model_linear_system_gaussian(config);
save(sprintf('%s/linear_system.mat', config.out_dir), 'linear_system');

%% learn a model
disp('Learning linear model');
model = struct();
num_features = size(linear_system,1);
model.w = (linear_system.A + config.gamma * eye(num_features)) \ linear_system.b;
%[B, fit_info] = lasso(linear_system.A, linear_system.b, 'CV', 10);
%lassoPlot(B,fit_info,'PlotType','CV');
%model.w = B(:,1);
%%
[model.sigma_tex, model.sigma_smooth] = learn_sigma_tex(model.w, config);

save(sprintf('%s/model.mat', config.out_dir), 'model');

%% training error
disp('Computing training error');
train_error = struct();
[train_error.D_nom_error, train_error.D_sq_error, train_D_pred] = ...
    depth_error_gaussian(config.training_nums, model, config);
save(sprintf('%s/training_error.mat', config.out_dir), 'train_error');

%% test error
disp('Computing test error');
test_error = struct();
[test_error.D_nom_error, test_error.D_sq_error, test_D_pred] = ...
    depth_error_gaussian(config.test_nums, model, config);
save(sprintf('%s/test_error.mat', config.out_dir), 'train_error');
