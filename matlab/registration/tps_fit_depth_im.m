function [tps, x_pts, y_pts, costs] = tps_fit_depth_im(D_x, D_y, u_x, u_y, K_x, K_y, w, config)
%TPS_FIT_DEPTH_IM Fits a thin plate spline to the image points u_x, u_y with
%corresponding depth measurements D_x and D_y and camera intrinsics K_{x,y}
%and with weights w

x_pts = project_depth_pinhole(D_x, u_x, K_x);
y_pts = project_depth_pinhole(D_y, u_y, K_y);

tps = tps_fit(x_pts, y_pts, config.bend_coef, config.rot_coef, w);

costs = tps_objective(tps, x_pts, y_pts);

end

