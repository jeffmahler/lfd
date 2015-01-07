% foolin around with gist
num_images = 849;
test_ind = round(rand() * (num_images-1) + 1);

data_dir = 'data/VOCB3DO';
rgb_dir = 'KinectColor';
rgb_name = 'img_%04d.png';
rgb_template = sprintf('%s/%s/%s', data_dir, rgb_dir, rgb_name);

ind = 1;
gist_features = zeros(512, num_images - 1);
train_images = cell(1,num_images-1);

for i = 0:(num_images-1)
    rgb_filename = sprintf(rgb_template, i);
    fprintf('Loading image %d\n', i);

    I = imread(rgb_filename);
    if i ~= test_ind
        gist_features(:,ind) = im2gist(I);
        train_images{ind} = I;
        ind = ind + 1;
    else
        test_image = I;
        test_gist = im2gist(I);
    end
end

kd = KDTreeSearcher(gist_features');

%% lookup and display results
K = 10;
indices = knnsearch(kd, test_gist', 'K', K);

figure;
subplot(1,K+1,1);
imshow(test_image);
title('Query image');

for j = 1:K
    subplot(1,K+1,j+1);
    imshow(train_images{indices(j)});
    title(sprintf('K = %d', j));
end

%% sift flow stuff
cellsize=3;
gridspacing=1;
winSize = 3;

I = test_image;
J = train_images{indices(1)};

I_small = double(imresize(I, 0.125));
J_small = double(imresize(J, 0.125));
sift1 = mexDenseSIFT(I_small,cellsize,gridspacing);
sift2 = mexDenseSIFT(J_small,cellsize,gridspacing);

SIFTflowpara.alpha=2*255;
SIFTflowpara.d=40*255;
SIFTflowpara.gamma=0.005*255;
SIFTflowpara.nlevels=3;
SIFTflowpara.wsize.x=5;
SIFTflowpara.wsize.y=5;
SIFTflowpara.topwsize.x=10;
SIFTflowpara.topwsize.y=3;
SIFTflowpara.nTopIterations = 60;
SIFTflowpara.nIterations= 30;

tic;[vx,vy,energylist]=SIFTflowc2f(sift1,sift2,SIFTflowpara);toc

%%
[X,Y]=meshgrid(1:80,1:60);
corr_x = X + vx;
corr_y = Y + vy;

% clean corrs
corr_x(corr_x < 1) = 1;
corr_y(corr_y < 1) = 1;
corr_x(corr_x > 80) = 80;
corr_y(corr_y > 60) = 60;

% invalid_corrs = corr_x < 1 | corr_x > 80 | corr_y < 1 | corr_y > 60;
% valid_corr_ind = find(invalid_corrs(:) == 0);

I_pts = [X(:), Y(:)];
J_pts = [corr_x(:), corr_y(:)];
num_corrs = size(J_pts, 1);

diff = zeros(num_corrs, 128);
for k = 1:num_corrs
    diff(k,:) = sift1(I_pts(k,2), I_pts(k,1), :) - ...
        sift2(J_pts(k,2), J_pts(k,1), :);
end
diff = sqrt(sum(diff.^2, 2));
norm_diff = (diff - min(diff)) / (max(diff) - min(diff));

total_corrs = 2000;
rand_indices = randsample(num_corrs, total_corrs);
corrs = {[I_pts(rand_indices,:) J_pts(rand_indices,:)]};
save('data/VOCB3DO/Features/siftflow_corrs.mat', 'corrs');
%%
figure(2);
imshow([uint8(I_small) uint8(J_small)]);

hold on;
corr_colors = zeros(num_corrs, 3);
corr_colors(:,1) = norm_diff;
for i = 1:num_corrs
    if norm_diff(i) > 0.75
        scatter(I_pts(i,1), I_pts(i,2), 'MarkerFaceColor', corr_colors(i,:), 'MarkerEdgeColor', corr_colors(i,:));
        scatter(J_pts(i,1) + 80, J_pts(i,2), 'MarkerFaceColor', corr_colors(i,:), 'MarkerEdgeColor', corr_colors(i,:));
        plot([I_pts(i,1); J_pts(i,1) + 80],...
             [I_pts(i,2); J_pts(i,2)], 'g');
    end
end

%%
[warpJ, mask]=warpImage(J_small,vx,vy);
figure(1);
subplot(1,4,1);
imshow(uint8(I_small));
subplot(1,4,2);
imshow(uint8(J_small));
subplot(1,4,3);
imshow(uint8(warpJ));

corr_conf_im = reshape(corr_colors, 60, 80, 3);
subplot(1,4,4);
imshow(corr_conf_im);


