function [I_filt_responses, I_gradients] = ...
    apply_filter_banks(image_pyr, filter_banks, config)
%APPLY_FILTER_BANKS Applys a series of filter banks in filter banks to
%  the 3 channels of the color image

show = config.show;
num_levels = image_pyr.num_levels;
num_filters = size(filter_banks, 2);
channels = 3;

I_filt_responses = cell(channels, num_levels);
I_gradients = cell(2, num_levels);

% loop through level, channels, and filters and apply each
for i = 1:num_levels
    [I_gradients{1, i}, I_gradients{2, i}] = ...
        imgradientxy(image_pyr.I_pyr{i}(:,:,1));
    for j = 1:channels
        cur_I = image_pyr.I_pyr{i}(:,:,j);
        for k = 1:num_filters
            if k == 1
                I_filt_responses{j, i} = ...
                    FbApply2d(cur_I, filter_banks{k}, 'same', show);
            else
                I_filt_responses{j, i} = ...
                    cat(3, I_filt_responses{j, i}, ...
                        FbApply2d(cur_I, filter_banks{k}, 'same', show));
            end
        end
    end
end

end

