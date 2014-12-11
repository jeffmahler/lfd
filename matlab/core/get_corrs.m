function [I_pts, J_pts] = get_corrs(I, J, scale)
%GET_CORRS Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    scale = 1.0;
end

[height, width, channels] = size(I);

% rescale images
I_rescaled = imresize(I, scale);
J_rescaled = imresize(J, scale);

% assumes images are the same size
figure(100);
imshow([I_rescaled, J_rescaled]);

disp('Click correspondences, left then right. Press the X key when finished');

% round up correspondences, subtract 1 because of matlab 1 indexing
I_pts = [];
J_pts = [];
term_sig = 0;
while term_sig ~= 120
    [left_x, left_y, term_sig] = ginput(1);
    if term_sig ~= 120
        I_pts = [I_pts; left_x, left_y];

        [right_x, right_y, term_sig] = ginput(1);
        J_pts = [J_pts; right_x - scale * width, right_y];
    end
end

% plot correspondences
num_corrs = size(I_pts, 1);
hold on;
for i = 1:num_corrs
    scatter(I_pts(i,1), I_pts(i,2), 'g');
    scatter(J_pts(i,1) + scale * width, J_pts(i,2), 'g');
    plot([I_pts(i,1); J_pts(i,1) + scale * width],...
         [I_pts(i,2); J_pts(i,2)], 'g');
end

I_pts = I_pts / scale;
J_pts = J_pts / scale;


