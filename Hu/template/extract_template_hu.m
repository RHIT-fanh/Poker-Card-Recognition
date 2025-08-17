clear;
close all;

img_path = 'diamond_flipped.png';                      % 输入图像路径
[~, base, ~] = fileparts(img_path);              % 文件名（无扩展名）

img_rgb = imread(img_path);                      % 读取图像

%% extractROI
gray  = rgb2gray(img_rgb);
blur  = imgaussfilt(gray, 1);
bin   = ~imbinarize(blur, 150/255);              % 白前景黑背景
clean = bwareaopen(bin, 300);                    % 移除小区域

figure; imshow(clean); title('Binary Clean Mask');

stats = regionprops(clean, 'BoundingBox', 'Area', 'Image');
valid_stats = stats([stats.Area] > 100);
fprintf('Detected %d candidate ROIs\n', numel(valid_stats));

%% Save each ROI's Hu moments as a separate file
output_dir = 'D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\try\template';                       % 输出目录
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

for k = 1:numel(valid_stats)
    roi = valid_stats(k).Image;                  % 获取 ROI 的二值图

    figure; imshow(roi); title(sprintf('ROI %d - Binary Region', k));

    hu = computeHuMoments(roi);                  % 计算 Hu moments
    fprintf('ROI %d Hu Moments:\n', k); disp(hu);

    % 构造保存路径：<图像名>_ROI_01_hu.mat
    save_name = sprintf('%s_ROI_%02d_hu.mat', base, k);
    save(fullfile(output_dir, save_name), 'hu'); % 仅保存 hu 一个变量
end

%% --- Hu moments 计算函数 ---
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
