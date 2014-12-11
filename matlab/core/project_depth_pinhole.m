function pts = project_depth_pinhole(D, u, K)
%PROJECT_DEPTH_PINHOLE Projects depths D (vector) at pixels u using intrinsics K

num_pix = size(D, 1);
pts_2d = [u, ones(num_pix, 1)]';

D_tiled = repmat(D, 1, 3);
D_tiled = D_tiled';

pts = D_tiled .* (inv(K) * pts_2d);
pts = pts';

end

