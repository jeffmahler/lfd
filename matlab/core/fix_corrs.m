function new_corrs = fix_corrs(corrs, height, width, level)
    num_pts = size(corrs, 1);
    new_corrs = zeros(0, 4);
    corrs = corrs / (2^(level-1));
    
    for i = 1:num_pts
        cur_corr = corrs(i,:);
        if sum(cur_corr < 1) == 0 && ...
                cur_corr(1) <= width && cur_corr(2) <= height && ...
                cur_corr(3) <= width && cur_corr(4) <= height
            new_corrs = [new_corrs; cur_corr];
        end
    end
end