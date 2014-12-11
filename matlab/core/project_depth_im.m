function pts = project_depth_im(D, K)
%PROJECT_DEPTH_IM Project the depth image D

[height, width] = size(D);
[X, Y] = meshgrid(1:width, 1:height);
pts_2d = [X(:), Y(:)];
pts_2d = pts_2d - 1; % convert to 0 indexing

pts = project_depth_pinhole(D(:), pts_2d, K);

end

