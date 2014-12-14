%% sift dense corrs
rgb_filename = sprintf('%s/%s/%s.png', data_dir, rgb_dir, 'img_0047');
I = imread(rgb_filename);
rgb2_filename = sprintf('%s/%s/%s.png', data_dir, rgb_dir, 'img_0046');
J = imread(rgb2_filename);

cellsize=3;
gridspacing=1;
winSize = 3;

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
% I_gray = double(rgb2gray(I_small));
% J_gray = double(rgb2gray(J_small));
[warpJ, mask]=warpImage(J_small,vx,vy);
figure(1);
subplot(1,3,1);
imshow(uint8(I_small));
subplot(1,3,2);
imshow(uint8(J_small));
subplot(1,3,3);
imshow(uint8(warpJ));

%%
[X,Y]=meshgrid(1:80,1:60);
corr_x = X + vx;
corr_y = Y + vy;
invalid_corrs = corr_x < 1 | corr_x > 80 | corr_y < 1 | corr_y > 60;
valid_corr_ind = find(invalid_corrs(:) == 0);

I_pts = [X(valid_corr_ind), Y(valid_corr_ind)];
J_pts = [corr_x(valid_corr_ind), corr_y(valid_corr_ind)];
num_corrs = size(J_pts, 1);

total_corrs = 2000;
rand_indices = randsample(num_corrs, total_corrs);
corrs = {[I_pts(rand_indices,:) J_pts(rand_indices,:)]};
save('data/VOCB3DO/Features/siftflow_corrs.mat', 'corrs');

figure(2);
imshow([uint8(I_small) uint8(J_small)]);

hold on;
for i = 1:51:num_corrs
    scatter(I_pts(i,1), I_pts(i,2), 'g');
    scatter(J_pts(i,1) + 80, J_pts(i,2), 'g');
    plot([I_pts(i,1); J_pts(i,1) + 80],...
         [I_pts(i,2); J_pts(i,2)], 'g');
end


