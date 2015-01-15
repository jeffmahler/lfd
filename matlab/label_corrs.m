function [matches, corrs] = label_corrs(img_nums, gist_db, config)

num_label = size(img_nums, 1);
matches = zeros(2, num_label);
corrs = cell(1, num_label);

for i = 1:num_label
    % get most similar image
    img_num = img_nums(i);
    [I, I_neighb, ind_neighb] = ...
        lookup_images_gist(img_num, gist_db, config.num_matches, config);
    
    for j = 1:config.num_matches
        J = I_neighb{j};
        figure(100);
        imshow([I, J]);
        pause(1.0);
    end
    
    result = input('Enter number of best match: ');
    J = I_neighb{result};
    ind = ind_neighb(result);
        
    % matches
    matches(1,i) = img_num;
    matches(2,i) = ind;
    
    if mod(i-1, config.out_rate) == 0
        fprintf('Labeling corrs for image %d\n', i);
        fprintf('Matched %d to %d in db\n', img_num, ind);
    end
    
    % get correspondences
    [train_pts, test_pts] = get_corrs(I, J, 1);
    corrs{i} = round([train_pts, test_pts]);
end

end

