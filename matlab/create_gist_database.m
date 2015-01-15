function gist_db = create_gist_database(config)
%CREATE_GIST_DATABASE Creates a kd tree on gist features for fast image
%similarity checking

% matrices to store intermediate data
num_training = size(config.training_nums, 1);

% loop through training set and 
for k = 1:num_training
    % read the next filename
    img_num = config.training_nums(k);
    rgb_filename = sprintf(config.rgb_file_template, img_num);
    
    if mod(k-1, config.out_rate) == 0
        fprintf('Processing image %d: %s\n', k, rgb_filename);
    end
    
    % read image and downsamples
    I = imread(rgb_filename);
    
    % compute gist representation
    I_gist_vec = im2gist(I);
    
    if ~exist('gist_vecs', 'var')
        gist_size = size(I_gist_vec, 1);
        gist_vecs = zeros(num_training, gist_size);
    end
    gist_vecs(k, :) = I_gist_vec';
end

gist_db = KDTreeSearcher(gist_vecs);

end

