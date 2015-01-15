function [img, neighbor_imgs, neighbor_nums] = ...
    lookup_images_gist(img_num, gist_db, K, config)
%LOOKUP_IMAGE_GIST

% read image and downsample
rgb_filename = sprintf(config.rgb_file_template, img_num);
I = imread(rgb_filename);

% compute gist and lookup neighbor indices
I_gist_vec = im2gist(I);
neighbor_indices = knnsearch(gist_db, I_gist_vec', 'K', K);
neighbor_nums = zeros(K, 1);

% load in the neighbors
neighbor_imgs = cell(1, K);
for i = 1:K
    neighbor_nums(i) = config.training_nums(neighbor_indices(i));
    rgb_filename = sprintf(config.rgb_file_template, neighbor_nums(i));
    neighbor_imgs{i} = imread(rgb_filename);
end
img = I;

end

