% scratch code for 2D TPS RPM

data_dir = 'data/VOCB3DO';
depth_dir = 'RegisteredDepthData';
rgb_dir = 'KinectColor';
img_name = 'img_0000';

rgb_filename = sprintf('%s/%s/%s.png', data_dir, rgb_dir, img_name);
depth_filename = sprintf('%s/%s/%s_abs_smooth.png', data_dir, depth_dir, img_name);

I = imread(rgb_filename);
D = imread(depth_filename);

figure(1);
subplot(1,2,1);
imshow(I);
title('RGB');
subplot(1,2,2);
imshow(histeq(D));
title('Depth');

%% create image pyramids
num_levels = 3;
start_level = 4;
I_pyr = cell(1, num_levels);
D_pyr = cell(1, num_levels);

I_pyr{1} = rgb2ycbcr(I);
D_pyr{1} = D;

for i = 1:(start_level-1)
   I_pyr{1} = impyramid(I_pyr{1}, 'reduce');
   D_pyr{1} = impyramid(D_pyr{1}, 'reduce');
end

for i = 2:num_levels
   I_pyr{i} = impyramid(I_pyr{i-1}, 'reduce');
   D_pyr{i} = impyramid(D_pyr{i-1}, 'reduce');
end

figure(2);
subplot(2,2,1);
imshow(I_pyr{3}(:,:,1));
title('Y');
subplot(2,2,2);
imshow(I_pyr{3}(:,:,2));
title('Cb');
subplot(2,2,3);
imshow(I_pyr{3}(:,:,3));
title('Cr');
subplot(2,2,4);
imshow(histeq(D));
title('Depth');

%% create custom filter banks
filt_size = 13.0;
LM_fb = makeLMfilters(filt_size); % Leung-Malik filter bank
S_fb = makeSfilters(filt_size);   % Schmid filter bank
LW_fb = makeLWfilters;            % Laws texture masks

num_LM_filts = size(LM_fb, 3);
num_S_filts = size(S_fb, 3);
num_LW_filts = size(LW_fb, 3);
sz_LW_filts = size(LW_fb, 1);

%% Piotor's filtering
% Flag codes
%   1: filter bank from Serge Belongie
%   2: 1st/2nd order DooG filters.  Similar to Gabor filterbank.
%   3: similar to Laptev&Lindberg ICPR04
%   4: decent seperable steerable? filterbank
%   5: berkeley filterbank for textons papers
%   6: symmetric DOOG filter
d = 2;
flag = 1;
show = 1;
serge_fb = FbMake(d, 1, show);
doog_fb = FbMake(d, 2, show);

num_serge_filts = size(serge_fb, 3);
num_doog_filts = size(doog_fb, 3);

%% apply filters
% for now just Laws and serge filter bank
show = 0;
I_filt_responses = cell(3, num_levels);
I_gradients = cell(2, num_levels);

for i = 1:num_levels
    [I_gradients{1, i}, I_gradients{2, i}] = ...
        imgradientxy(I_pyr{i}(:,:,1));
    for j = 1:3
        cur_I = I_pyr{i}(:,:,j);
        I_filt_responses{j, i} = ...
             FbApply2d(cur_I, serge_fb, 'same', show);
        I_filt_responses{j, i} = ...
             cat(3, I_filt_responses{j, i}, ...
                 FbApply2d(cur_I, LW_fb, 'same', show));
    end
end

%% create feature rep
reg_level = 1;
intensity_channel = 1;
num_channels = 3;
num_neighbors = 5;

[height, width, num_filts] = size(I_filt_responses{intensity_channel, reg_level});
num_pix = height * width;

num_features = num_filts * num_levels * num_neighbors * num_channels;
Phi = zeros(num_pix, num_features);
start_I = 1;
end_I = start_I + num_filts - 1;

for i = 1:num_levels
    fprintf('Accumulating features for level %d\n', i);
    for j = 1:num_channels
    fprintf('Accumulating features for channel %d\n', j);
    
    filt_resp = I_filt_responses{j, i};
    %[height, width, num_filts] = size(filt_resp);
    
    % center
    filt_resp_big = imresize(filt_resp, 2^(i-1), 'nearest');
    Phi(:, start_I:end_I) = ...
        reshape(filt_resp_big, height*width, num_filts);
    start_I = start_I + num_filts;
    end_I = start_I + num_filts - 1;
    
    % down
    H = vision.GeometricTranslator;
    H.OutputSize = 'Same as input image';
    H.Offset = [-1, 0];
    filt_resp_tr = step(H, filt_resp);
    filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
    Phi(:, start_I:end_I) = ...
        reshape(filt_resp_big, height*width, num_filts);
    start_I = start_I + num_filts;
    end_I = start_I + num_filts - 1;
    
    % up
    H = vision.GeometricTranslator;
    H.OutputSize = 'Same as input image';
    H.Offset = [1, 0];
    filt_resp_tr = step(H, filt_resp);
    filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
    Phi(:, start_I:end_I) = ...
        reshape(filt_resp_big, height*width, num_filts);
    start_I = start_I + num_filts;
    end_I = start_I + num_filts - 1;
    
    % right
    H = vision.GeometricTranslator;
    H.OutputSize = 'Same as input image';
    H.Offset = [0, -1];
    filt_resp_tr = step(H, filt_resp);
    filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
    Phi(:, start_I:end_I) = ...
        reshape(filt_resp_big, height*width, num_filts);
    start_I = start_I + num_filts;
    end_I = start_I + num_filts - 1;
    
    % left
    H = vision.GeometricTranslator;
    H.OutputSize = 'Same as input image';
    H.Offset = [0, 1];
    filt_resp_tr = step(H, filt_resp);
    filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
    Phi(:, start_I:end_I) = ...
        reshape(filt_resp_big, height*width, num_filts);
    start_I = start_I + num_filts;
    end_I = start_I + num_filts - 1;
    end
end

%% check depth difference histograms
use_inv_depth = true;

diff_right_filter = ...
    [0, 0, 0;
    -1, 1, 0;
    0, 0, 0];
diff_left_filter = ...
    [0, 0, 0;
    0, 1,-1;
    0, 0, 0];
diff_down_filter = ...
    [0, -1, 0;
    0, 1, 0;
    0, 0, 0];
diff_up_filter = ...
    [0, 0, 0;
    0, 1, 0;
    0, -1, 0];

weights = zeros(num_pix, 4);
D_diff = double(D_pyr{reg_level});
max_depth = max(max(D_diff));

if use_inv_depth
   D_diff = max_depth ./ D_diff; 
end

grad_scale = 1;
diff_D_weights_x = 1.0 ./ (1.0 + exp(grad_scale * abs(I_gradients{1, reg_level})));
diff_D_weights_y = 1.0 ./ (1.0 + exp(grad_scale * abs(I_gradients{2, reg_level})));
weights(:,1) = diff_D_weights_x(:);
weights(:,2) = diff_D_weights_x(:);
weights(:,3) = diff_D_weights_y(:);
weights(:,4) = diff_D_weights_y(:);
weights(weights < 1e-4) = 0;

diff_D_left = conv2(D_diff, diff_left_filter, 'same');
diff_D_right = conv2(D_diff, diff_right_filter, 'same');
diff_D_up = conv2(D_diff, diff_up_filter, 'same');
diff_D_down = conv2(D_diff, diff_down_filter, 'same');

diff_D_left_weighted = diff_D_weights_x .* diff_D_left.^2;
diff_D_right_weighted = diff_D_weights_x .* diff_D_right.^2;
diff_D_up_weighted = diff_D_weights_y .* diff_D_up.^2;
diff_D_down_weighted = diff_D_weights_y .* diff_D_down.^2;

sigma_smooth = 0;
sigma_smooth = sigma_smooth + sum(sum(diff_D_left_weighted));
sigma_smooth = sigma_smooth + sum(sum(diff_D_right_weighted));
sigma_smooth = sigma_smooth + sum(sum(diff_D_up_weighted));
sigma_smooth = sigma_smooth + sum(sum(diff_D_down_weighted));
sigma_smooth = sigma_smooth / (4 * num_pix);

% num_bins = 100;
% figure(5);
% subplot(1,2,1);
% hist(diff_D_left(:), 1000);
% subplot(1,2,2);
% hist(diff_D_left_weighted(:), 1000);

%% regress to depth
gamma = 1e-2;
d_vec_train = double(D_pyr{reg_level}(:));
d_vec = d_vec_train;
if use_inv_depth
    d_vec = max_depth ./ d_vec_train;
end
w = (Phi' * Phi + gamma * eye(num_features)) \ (Phi' * d_vec);

% predict depth for training image 
d_vec_pred = Phi * w;
d_sq_error = (d_vec_train - d_vec_pred).^2;
if use_inv_depth
    d_vec_inv_train = max_depth ./ d_vec_train;   
    d_sq_error = (d_vec_inv_train - d_vec_pred).^2;
end
sigma_tex = mean(d_sq_error);

% TODO: replace with QP solver
%%
%d_vec_pred = g;
if use_inv_depth
    d_vec_pred = max_depth ./ d_vec_pred;
end

d_sq_error = (d_vec_train - d_vec_pred).^2;
d_nom_error = d_vec_train - d_vec_pred;

n_bins = 100;
figure(10);
hist(d_nom_error, n_bins);
title('Depth Error');

fprintf('Nominal Error\n');
fprintf('Mean:\t%.03f\n', mean(d_nom_error));
fprintf('Med:\t%.03f\n', median(d_nom_error));
fprintf('Std:\t%.03f\n', std(d_nom_error));

fprintf('\nSquared Error\n');
fprintf('Mean:\t%.03f\n', mean(d_sq_error));
fprintf('Med:\t%.03f\n', median(d_sq_error));
fprintf('Min:\t%.03f\n', min(d_sq_error));
fprintf('Max:\t%.03f\n', max(d_sq_error));

D_pred = reshape(d_vec_pred, height, width);
figure(11);
subplot(1,2,1);
imshow(D_pyr{reg_level});
subplot(1,2,2);
imshow(uint16(D_pred));

%% form linear system
%sigma_tex = 1.0;
%sigma_smooth = 1.0;
lambda = sigma_tex / sigma_smooth;

% temp weights matrix organized as (L R U D)
%weights = ones(num_pix, 4);

A = zeros(num_pix);
b = Phi * w;

% update off-diagonal
im_size = [height width];
mat_size = size(A);
lin_ind = 1:num_pix;
[px_y, px_x] = ind2sub(im_size, lin_ind);
ind_left = [max([px_x - 1; ones(1, num_pix)]); px_y]; %max(px_x - 1, ones(1, num_pix));
ind_right = [min([px_x + 1; width*ones(1, num_pix)]); px_y];
ind_up = [px_x; max([px_y - 1; ones(1, num_pix)])];
ind_down = [px_x; min([px_y + 1; height*ones(1, num_pix)])];

% compute linear indices
lin_ind_left = sub2ind(im_size, ind_left(2,:), ind_left(1,:));
lin_ind_right = sub2ind(im_size, ind_right(2,:), ind_right(1,:));
lin_ind_up = sub2ind(im_size, ind_up(2,:), ind_up(1,:));
lin_ind_down = sub2ind(im_size, ind_down(2,:), ind_down(1,:));
lin_ind_diag = 1:(num_pix+1):(num_pix*num_pix);

% fill left symmetric
lin_ind_fill = sub2ind(mat_size, lin_ind, lin_ind_left);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda * weights(:,1);
lin_ind_fill = sub2ind(mat_size, lin_ind_left, lin_ind);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda * weights(:,1);

% fill right symmetric
lin_ind_fill = sub2ind(mat_size, lin_ind, lin_ind_right);
%A(lin_ind_fill) = -2 * lambda * weights(:,2);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda * weights(:,2);
lin_ind_fill = sub2ind(mat_size, lin_ind_right, lin_ind);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda * weights(:,2);

% fill up symmetric
lin_ind_fill = sub2ind(mat_size, lin_ind, lin_ind_up);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda * weights(:,3);
lin_ind_fill = sub2ind(mat_size, lin_ind_up, lin_ind);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda * weights(:,3);

% % fill down symmetric
lin_ind_fill = sub2ind(mat_size, lin_ind, lin_ind_down);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda * weights(:,4);
lin_ind_fill = sub2ind(mat_size, lin_ind_down, lin_ind);
A(lin_ind_fill) = A(lin_ind_fill)' - lambda * weights(:,4);

% update diagonal
A(lin_ind_diag) = ones(num_pix,1) + lambda * sum(weights, 2);

lin_ind_fill = sub2ind(mat_size, lin_ind_left, lin_ind_left);
A(lin_ind_fill) = A(lin_ind_fill)' + lambda * weights(:,1);

lin_ind_fill = sub2ind(mat_size, lin_ind_right, lin_ind_right);
A(lin_ind_fill) = A(lin_ind_fill)' + lambda * weights(:,2);

lin_ind_fill = sub2ind(mat_size, lin_ind_up, lin_ind_up);
A(lin_ind_fill) = A(lin_ind_fill)' + lambda * weights(:,3);

lin_ind_fill = sub2ind(mat_size, lin_ind_down, lin_ind_down);
A(lin_ind_fill) = A(lin_ind_fill)' + lambda * weights(:,4);

%A(lin_ind_diag) = ones(num_pix,1) + 2 * lambda * sum(weights, 2);

%% solve linear system
d_vec_pred = A \ b;
g = d_vec_pred;

%% junk code below

% fucking around with translation
im = I_pyr{3}(:,:,1);

H = vision.GeometricTranslator;
H.Offset = [1, 0];
H.OutputSize = 'Same as input image';
im_trans = step(H, im);
a = imresize(im_trans, 2.0, 'nearest');
figure;
subplot(1,2,1);
imshow(im);
subplot(1,2,2);
imshow(im_trans);

%
I_serge_filt = FbApply2d(I_pyr{3}(:,:,1), serge_fb, 'same', show);
I_LW_filt = FbApply2d(I_pyr{3}(:,:,1), LW_fb, 'same', show);
K = cat(3, I_serge_filt, I_LW_filt);

total_num_filts = num_LM_filts + num_S_filts + num_LW_filts;

%% apply filters

I_filt_responses = cell(total_num_filts, num_levels);

for i = 1:num_levels
    filt_index = 1;
    for j = 1:num_LM_filts
     I_filt_responses{filt_index, i} = ...
         imfilter(I_pyr{i}, LM_fb(:,:,j), 'replicate');
     filt_index = filt_index + 1;
    end
    for j = 1:num_S_filts
     I_filt_responses{filt_index, i} = ...
         imfilter(I_pyr{i}, S_fb(:,:,j), 'replicate');
     filt_index = filt_index + 1;
    end
    for j = 1:num_LW_filts
     I_filt_responses{filt_index, i} = ...
         imfilter(I_pyr{i}, LW_fb(:,:,j), 'replicate');
     filt_index = filt_index + 1;
    end
end

%%
filt = 39;
level = 2;

figure(2);
subplot(1,2,1);
imagesc(LM_fb(:,:,filt));
subplot(1,2,2);
imshow(I_filt_responses{filt,level}(:,:,1));

%%
% num_LM_filts = size(LM_filts_big, 3);
% num_S_filts = size(S_filts_big, 3);
% LM_big_filt_size = size(LM_filts_big, 1);
% S_big_filt_size = size(S_filts_big, 1);
% 
% LM_filts = zeros(filt_size, filt_size, num_LM_filts);
% S_filts = zeros(filt_size, filt_size, num_S_filts);
% 
% for i = 1:num_LM_filts
%     LM_filts(:,:,i) = imresize(LM_filts_big(:,:,i), double(filt_size) / LM_big_filt_size);
% end
% 
% for i = 1:num_S_filts
%     S_filts(:,:,i) = imresize(S_filts_big(:,:,i), double(filt_size) / S_big_filt_size);
% end
