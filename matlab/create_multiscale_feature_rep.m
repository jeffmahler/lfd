function Phi = create_multiscale_feature_rep(I_filt_responses, config)
%CREATE_MULTISCALE_FEATURE_REP Compiles a list of features for each pixel
%  for each image in an image pyramid

% global constants
base_level = 1;
num_channels = 3;
num_neighbors = 5;
intensity_channel = 1;
num_levels = size(I_filt_responses, 2);

% read in useful data
[height, width, num_filts] = ...
    size(I_filt_responses{intensity_channel, base_level});
num_pix = height * width;
num_features = num_filts * num_levels * num_neighbors * num_channels;

Phi = zeros(num_pix, num_features);

% indices for adding to the matrix
start_I = 1;
end_I = start_I + num_filts - 1;

for i = 1:num_levels
    if config.debug
        %fprintf('Accumulating features for level %d\n', i);
    end
    for j = 1:num_channels
        if config.debug
            %fprintf('Accumulating features for channel %d\n', j);
        end
        filt_resp = I_filt_responses{j, i};
        [lev_height, lev_width, ~] = size(filt_resp);
        
        % center
        filt_resp_big = imresize(filt_resp, 2^(i-1), 'nearest');
        Phi(:, start_I:end_I) = ...
            reshape(filt_resp_big, height*width, num_filts);
        start_I = start_I + num_filts;
        end_I = start_I + num_filts - 1;

        % down
        filt_resp_tr = zeros(lev_height, lev_width, num_filts);
        filt_resp_tr(1:(lev_height-1),:,:) = filt_resp(2:lev_height,:,:);
        filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
        Phi(:, start_I:end_I) = ...
            reshape(filt_resp_big, height*width, num_filts);
        start_I = start_I + num_filts;
        end_I = start_I + num_filts - 1;
        
        % up
        filt_resp_tr = zeros(lev_height, lev_width, num_filts);
        filt_resp_tr(2:lev_height,:,:) = filt_resp(1:(lev_height-1),:,:);
        filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
        Phi(:, start_I:end_I) = ...
            reshape(filt_resp_big, height*width, num_filts);
        start_I = start_I + num_filts;
        end_I = start_I + num_filts - 1;

        % right
        filt_resp_tr = zeros(lev_height, lev_width, num_filts);
        filt_resp_tr(:,1:(lev_width-1),:) = filt_resp(:,2:lev_width,:);
        filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
        Phi(:, start_I:end_I) = ...
            reshape(filt_resp_big, height*width, num_filts);
        start_I = start_I + num_filts;
        end_I = start_I + num_filts - 1;

        % left
        filt_resp_tr = zeros(lev_height, lev_width, num_filts);
        filt_resp_tr(:,2:lev_width,:) = filt_resp(:,1:(lev_width-1),:);
        filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
        Phi(:, start_I:end_I) = ...
            reshape(filt_resp_big, height*width, num_filts);
        start_I = start_I + num_filts;
        end_I = start_I + num_filts - 1;
    end
end

end

