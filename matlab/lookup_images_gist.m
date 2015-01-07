function [img, neighbor_imgs, neighbor_indices] = ...
    lookup_images_gist(img_num, gist_db, K, config)
%LOOKUP_IMAGE_GIST

% read image and downsample
rgb_filename = sprintf(config.rgb_file_template, img_num);
I = imread(rgb_filename);
for i = 1:(config.start_level-1)
   I = impyramid(I, 'reduce');
end

% compute gist and lookup neighbor indices
I_gist_vec = im2gist(I);
neighbor_indices = knnsearch(gist_db, I_gist_vec', 'K', K);

% load in the neighbors
neighbor_imgs = cell(1, K);
for i = 1:K
    rgb_filename = sprintf(config.rgb_file_template, neighbor_indices(i));
    neighbor_imgs{i} = imread(rgb_filename);
    for j = 1:(config.start_level-1)
       neighbor_imgs{i} = impyramid(neighbor_imgs{i}, 'reduce');
    end
end
img = I;

end

