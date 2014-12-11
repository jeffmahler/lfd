function [Phi, I_gradients] = ...
    extract_texture_features(image_pyr, filter_banks, config)
%EXTRACT_FEATURES Get the features 

start_time = tic;
[I_filt_responses, I_gradients] = ...
    apply_filter_banks(image_pyr, filter_banks, config);
duration = toc(start_time);
if config.debug
    %fprintf('Filter banks took %f sec\n', duration);
end

if config.use_texture_energy
    I_filt_responses = compute_texture_energy(I_filt_responses, config);
end

start_time = tic;
Phi = create_multiscale_feature_rep(I_filt_responses, config);
duration = toc(start_time);
if config.debug
    %fprintf('Multiscale rep took %f sec\n', duration);
end

Phi = [Phi, ones(size(Phi,1),1)]; % add constant to each

end

