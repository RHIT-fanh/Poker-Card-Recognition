% clc; 
clear; 
% close all;

[filename, pathname] = uigetfile({ ...
    '*.jpg;*.png;*.bmp;*.tif;*.tiff', 'Image files (*.jpg, *.png, *.bmp, *.tif, *.tiff)'; ...
    '*.*', 'All Files (*.*)'}, ...
    'Select image file');


if isequal(filename, 0)
    disp('User canceled file selection.');
    return;
end
img_path = fullfile(pathname, filename);
img = imread(img_path);

% === Step 2: Convert to binary image with white foreground and black background ===
if size(img, 3) == 3
    gray = rgb2gray(img);
else
    gray = img;
end

% Check if it's already binary
if islogical(gray) || all(unique(gray) <= 1)
    bw = logical(gray);     % Already binary
else
    bw = imbinarize(gray);  % Auto-threshold
end


% Display binary image
figure('Name', 'Processed Binary Image');
imshow(bw); title('Binary Image (White Foreground)');

% === Step 3: Compute Hu Moments ===
hu = computeHuMoments(bw);

% === Step 4: Print result ===
fprintf('\nFile: %s\n', filename);
fprintf('Hu Moments:\n');
disp(hu);

% === Hu moment computation function ===
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
