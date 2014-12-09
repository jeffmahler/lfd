function image_pyr = load_image_pyramid(rgb_filename, depth_filename, config)
%LOAD_IMAGE_PYRAMID Load in an image pyramid from the VOCB3DO dataset using
% the specified configuration file which contains
%   num_levels- number of image pyramid levels to store
%   start_level- level at which to start building the pyramid
%   level_rate - specifies the granularity of levels (eg. 2 = every other)
%   color_space- specifies the color space to use
%   vis - whether or not to visualize the outputs

% read in images
I = imread(rgb_filename);
D = imread(depth_filename);

if config.vis
    figure(1);
    subplot(1,2,1);
    imshow(I);
    title('RGB');
    subplot(1,2,2);
    imshow(histeq(D));
    title('Depth');
end

% create image pyramids
I_pyr = cell(1, config.num_levels);
D_pyr = cell(1, config.num_levels);

if strcmp(config.color_space, 'hsv')
    I_pyr{1} = rgb2hsv(I);
elseif strcmp(config.color_space, 'ycbcr')
    I_pyr{1} = rgb2ycbcr(I);
else
    I_pyr{1} = I;
end
D_pyr{1} = D;

% get to the starting level
for i = 1:(config.start_level-1)
   I_pyr{1} = impyramid(I_pyr{1}, 'reduce');
   D_pyr{1} = impyramid(D_pyr{1}, 'reduce');
end

% form the pyramid
k = 1;
i = 2;
while i <= config.num_levels
    if mod(k, config.level_rate) == 0
        I_pyr{i} = impyramid(I_pyr{i-1}, 'reduce');
        D_pyr{i} = impyramid(D_pyr{i-1}, 'reduce');
        i = i+1;
    else % in place reduction
        I_pyr{i} = impyramid(I_pyr{i}, 'reduce');
        D_pyr{i} = impyramid(D_pyr{i}, 'reduce');
    end
    k = k+1;
end

if config.vis
    figure(2);
    subplot(2,2,1);
    imshow(I_pyr{3}(:,:,1));
    title('Channel 1');
    subplot(2,2,2);
    imshow(I_pyr{3}(:,:,2));
    title('Channel 2');
    subplot(2,2,3);
    imshow(I_pyr{3}(:,:,3));
    title('Channel 3');
    subplot(2,2,4);
    imshow(histeq(D));
    title('Depth');
end

% form outpur struct
image_pyr = struct();
image_pyr.I_pyr = I_pyr;
image_pyr.D_pyr = D_pyr;
image_pyr.start_level = config.start_level;
image_pyr.num_levels = size(I_pyr,2);
image_pyr.im_height = size(D_pyr{1}, 1);
image_pyr.im_width = size(D_pyr{1}, 2);
image_pyr.max_depth = max(max(D_pyr{1}));

end