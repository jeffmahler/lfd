function [matches, corrs] = label_corrs(img_nums, gist_db, config)

num_label = size(img_nums, 1);
matches = zeros(2, num_label);
corrs = cell(1, num_label);

for i = 1:num_label
    % get most similar image
    img_num = img_nums(i);
    [I, I_neighb, ind_neighb] = lookup_images_gist(img_num, gist_db, 1, config);
    J = I_neighb{1};
    ind = ind_neighb(1);
    
    % matches
    matches(1,i) = img_num;
    matches(2,i) = ind;
    
    if mod(i-1, config.out_rate) == 0
        fprintf('Labeling corrs for image %d: img_%d\n', i, img_num);
    end
    
    % get correspondences
    [train_pts, test_pts] = get_corrs(I, J, config.corr_scale);
    corrs{k} = round([train_pts, test_pts]);
end

end

