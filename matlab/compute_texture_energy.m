function I_tex_energy = compute_texture_energy(I_filt_responses, config)
%COMPUTE_TEXTURE_ENERGY Computes the abs, energy, and kurtosis for filters

patch_size_x = config.patch_size_x;
patch_size_y = config.patch_size_y;
[num_channels, num_levels] = size(I_filt_responses);

% create averaging filter
avg_filter = ones(2*patch_size_x+1, 2*patch_size_y+1);
avg_filter = avg_filter / ((2*patch_size_x+1) * (2*patch_size_y+1));

% compute num energy opts
num_energy = 1 + config.use_energy + config.use_kurtosis;

I_tex_energy = cell(num_channels, num_levels);
for i = 1:num_levels
    for j = 1:num_channels
        I_response = I_filt_responses{j, i};
        [height, width, num_features] = size(I_response);
        num_energy_features = num_features * num_energy;
        I_energy = zeros(height, width, num_energy_features);
        
        index = 1;
        for k = 1:num_features
            abs_I_response = abs(I_response(:,:,k)); % absolute value of response
        
            % compute sum of powers of abs values over a window
            I_energy(:,:,index) = conv2(abs_I_response, avg_filter, 'same');
            index = index + 1;
            
            % compute energy (sum of abs sq)
            if config.use_energy
                eng_I_response = abs_I_response.^2;
                I_energy(:,:,index) = conv2(eng_I_response, avg_filter, 'same');
                index = index + 1;
            end

            % compute kurtosis (sum of abs to the fourth)
            if config.use_kurtosis
                kur_I_response = abs_I_response.^4;
                I_energy(:,:,index) = conv2(kur_I_response, avg_filter, 'same');
                index = index + 1;
            end
        end
        I_tex_energy{j,i} = I_energy;
    end
end

end

