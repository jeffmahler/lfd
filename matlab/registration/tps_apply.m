function y_t = tps_apply(tps, y)
%TPS_APPLY apply tps function to y

K_yx = tps_kernel_matrix(y, tps.x);

y_t = K_yx * tps.w_ng + y * tps.lin_ag + repmat(tps.trans_g, size(y,1), 1);

end

