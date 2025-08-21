clc; clear; close all;

img_path = '2.png';     
original = imread(img_path);
if size(original, 3) == 3
    original = rgb2gray(original);  
end
bw = imbinarize(original);

angles = 0:30:330;
hu_all = zeros(length(angles), 7);

figure('Name', 'Rotation vs Hu Moments');
for i = 1:length(angles)
    angle = angles(i);
    
    rotated = imrotate(bw, angle, 'bilinear', 'crop');
    
    hu = computeHuMoments(rotated);
    hu_all(i, :) = hu;
    
    subplot(3, ceil(length(angles)/3), i);
    imshow(rotated);
    title([num2str(angle) '°']);
end


disp('Rotation angle vs Hu moments:');
disp(array2table(hu_all, 'VariableNames', ...
    {'Hu1','Hu2','Hu3','Hu4','Hu5','Hu6','Hu7'}, ...
    'RowNames', cellstr(string(angles') + ' deg')));


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
