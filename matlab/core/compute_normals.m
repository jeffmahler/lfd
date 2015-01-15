function [surf_norms, mags] = compute_normals( image_pyr, config )
%COMPUTE_NORMALS Summary of this function goes here
%   Detailed explanation goes here
height = image_pyr.im_height;
width = image_pyr.im_width;
num_pix = height * width;
hund_um_to_m = config.hund_um_to_m;

D_im = hund_um_to_m * double(image_pyr.D_pyr{1});
pts = project_depth_im(D_im, image_pyr.K);
pts_grid = reshape(pts, [height, width, 3]);

dx = [-0.5 0 0.5];
dy = dx';

pts_dx_x = conv2(padarray(pts_grid(:,:,1), [0 1], 'replicate'), dx, 'valid');
pts_dx_y = conv2(padarray(pts_grid(:,:,2), [0 1], 'replicate'), dx, 'valid');
pts_dx_z = conv2(padarray(pts_grid(:,:,3), [0 1], 'replicate'), dx, 'valid');

pts_dy_x = conv2(padarray(pts_grid(:,:,1), [1 0], 'replicate'), dy, 'valid');
pts_dy_y = conv2(padarray(pts_grid(:,:,2), [1 0], 'replicate'), dy, 'valid');
pts_dy_z = conv2(padarray(pts_grid(:,:,3), [1 0], 'replicate'), dy, 'valid');

tan_x = [pts_dx_x(:) pts_dx_y(:) pts_dx_z(:)];
tan_y = [pts_dy_x(:) pts_dy_y(:) pts_dy_z(:)];

surf_grads = cross(tan_y', tan_x');
mags = sqrt(sum(surf_grads.^2, 1));
surf_grads(:,mags == 0) = -1.0 / sqrt(3); % set invalids to point down optical axis
mags(mags == 0 ) = 1.0;
mags_mat = repmat(mags, [3, 1]);
surf_norms = surf_grads ./ mags_mat;
surf_norms = reshape(surf_norms, [height, width, 3]);

end

