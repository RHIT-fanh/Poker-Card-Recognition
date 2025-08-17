clc; clear; close all;

[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp;*.tif;*.tiff','Image Files'}, 'Select an image');
if isequal(filename, 0)
    disp('User cancelled.'); return;
end
img = imread(fullfile(pathname, filename));
if size(img, 3) == 3
    img = rgb2gray(img);
end

bw = imbinarize(img);


scale_factors = 0.5:0.2:2.0;
n = numel(scale_factors);
hu_all = zeros(n, 7);

figure('Name', 'Stretched Images');
for i = 1:n
    scale_x = scale_factors(i);  % Stretch in X only
    scale_y = 1.0;               % No stretch in Y

    tform = affine2d([scale_x 0 0; 0 scale_y 0; 0 0 1]);
    stretched = imwarp(bw, tform);

    hu = computeHuMoments(stretched);
    hu_all(i, :) = hu;

    subplot(2, ceil(n/2), i);
    imshow(stretched); title(['X Scale = ' num2str(scale_x)]);
end

disp('X Scale vs Hu Moments:');
disp(array2table(hu_all, 'VariableNames', ...
    {'Hu1','Hu2','Hu3','Hu4','Hu5','Hu6','Hu7'}, ...
    'RowNames', cellstr("x" + string(scale_factors))));



function hu = computeHuMoments(BW)
    BW = BW > 0;
    [h, w] = size(BW);
    [X, Y] = meshgrid(1:w, 1:h);
    BWd = double(BW);
    m00 = sum(BWd(:));
    if m00 == 0
        hu = zeros(1, 7); return;
    end
    xc = sum(sum(X .* BWd)) / m00;
    yc = sum(sum(Y .* BWd)) / m00;
    mu = @(p,q) sum(sum(((X - xc).^p) .* ((Y - yc).^q) .* BWd));
    eta = @(p,q) mu(p,q) / m00^((p + q) / 2 + 1);
    n20 = eta(2,0); n02 = eta(0,2); n11 = eta(1,1);
    n30 = eta(3,0); n12 = eta(1,2); n21 = eta(2,1); n03 = eta(0,3);
    hu = [n20 + n02, ...
         (n20 - n02)^2 + 4 * n11^2, ...
         (n30 - 3*n12)^2 + (3*n21 - n03)^2, ...
         (n30 + n12)^2 + (n21 + n03)^2, ...
         (n30 - 3*n12)*(n30 + n12)*((n30 + n12)^2 - 3*(n21 + n03)^2) + ...
         (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)^2 - (n21 + n03)^2), ...
         (n20 - n02)*((n30 + n12)^2 - (n21 + n03)^2) + ...
         4 * n11 * (n30 + n12) * (n21 + n03), ...
         (3*n21 - n03)*(n30 + n12)*((n30 + n12)^2 - 3*(n21 + n03)^2) - ...
         (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)^2 - (n21 + n03)^2)];
end
