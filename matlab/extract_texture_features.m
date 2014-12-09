function [Phi, I_gradients] = ...
    extract_texture_features(image_pyr, filter_banks, config)
%EXTRACT_FEATURES Get the features 

[I_filt_responses, I_gradients] = ...
    apply_filter_banks(image_pyr, filter_banks, config);
Phi = create_multiscale_feature_rep(I_filt_responses, config);
Phi = [Phi, ones(size(Phi,1),1)]; % add constant to each

end

