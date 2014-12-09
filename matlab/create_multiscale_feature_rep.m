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

        % center
        filt_resp_big = imresize(filt_resp, 2^(i-1), 'nearest');
        Phi(:, start_I:end_I) = ...
            reshape(filt_resp_big, height*width, num_filts);
        start_I = start_I + num_filts;
        end_I = start_I + num_filts - 1;

        % down
        H = vision.GeometricTranslator;
        H.OutputSize = 'Same as input image';
        H.Offset = [-1, 0];
        filt_resp_tr = step(H, filt_resp);
        filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
        Phi(:, start_I:end_I) = ...
            reshape(filt_resp_big, height*width, num_filts);
        start_I = start_I + num_filts;
        end_I = start_I + num_filts - 1;

        % up
        H = vision.GeometricTranslator;
        H.OutputSize = 'Same as input image';
        H.Offset = [1, 0];
        filt_resp_tr = step(H, filt_resp);
        filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
        Phi(:, start_I:end_I) = ...
            reshape(filt_resp_big, height*width, num_filts);
        start_I = start_I + num_filts;
        end_I = start_I + num_filts - 1;

        % right
        H = vision.GeometricTranslator;
        H.OutputSize = 'Same as input image';
        H.Offset = [0, -1];
        filt_resp_tr = step(H, filt_resp);
        filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
        Phi(:, start_I:end_I) = ...
            reshape(filt_resp_big, height*width, num_filts);
        start_I = start_I + num_filts;
        end_I = start_I + num_filts - 1;

        % left
        H = vision.GeometricTranslator;
        H.OutputSize = 'Same as input image';
        H.Offset = [0, 1];
        filt_resp_tr = step(H, filt_resp);
        filt_resp_big = imresize(filt_resp_tr, 2^(i-1), 'nearest');
        Phi(:, start_I:end_I) = ...
            reshape(filt_resp_big, height*width, num_filts);
        start_I = start_I + num_filts;
        end_I = start_I + num_filts - 1;
    end
end

end

