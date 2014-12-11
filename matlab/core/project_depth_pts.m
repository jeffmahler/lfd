function [D, valid_mask] = project_depth_pts(pts, height, width, K)
%PROJECT_DEPTH_PTS Create a depth image by projecting the points in pts

D = realmax * ones(height, width);
depths = pts(:,3)';
pts_proj = K * pts';
pts_norm = pts_proj(1:2,:) ./ repmat(depths, 2, 1);
pts_norm_2d = round(pts_norm + 1);

% project all points into the image
num_pts = size(pts_norm_2d, 2);
for i = 1:num_pts
    u = pts_norm_2d(:,i);
    d = depths(i);
    if u(1) >= 1 && u(2) >= 1 && u(1) <= width && u(2) <= height
        D(u(2),u(1)) = min(d, D(u(2),u(1)));
    end
end

D(D == realmax) = 0;
D(D <= 0) = 0;
valid_mask = D > 0;

end

