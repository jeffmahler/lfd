function [left_pts, right_pts] = get_corrs(I, J)
%GET_CORRS Summary of this function goes here
%   Detailed explanation goes here

[height, width, channels] = size(I);

% assumes images are the same size
figure(100);
imshow([I, J]);

disp('Click correspondences, left then right. Press the X key when finished');

left_pts = [];
right_pts = [];
term_sig = 0;
while term_sig ~= 120
    [left_x, left_y, term_sig] = ginput(1);
    if term_sig ~= 120
        left_pts = [left_pts; left_x, left_y];

        [right_x, right_y, term_sig] = ginput(1);
        right_pts = [right_pts; right_x - width, right_y];
    end
end

% plot correspondences
num_corrs = size(left_pts, 1);
hold on;
for i = 1:num_corrs
    scatter(left_pts(i,1), left_pts(i,2), 'g');
    scatter(right_pts(i,1)+width, right_pts(i,2), 'g');
    plot([left_pts(i,1); right_pts(i,1)+width],...
         [left_pts(i,2); right_pts(i,2)], 'g');
end


