clc; clear; close all;

% === Step 1: 读取图像并灰度化 ===
img = imread('test.tif');   % 支持彩色或灰度图像

% === Step 2: 二值化 + 查找最大轮廓 ===
bw = img;
bw = imcomplement(bw);  % 白前景

% 查找轮廓
bw = bwareafilt(bw, 1);  % 保留最大连通域
stats = regionprops(bw, 'BoundingBox', 'ConvexHull');

% 获取角点
hull = stats.ConvexHull;  % N×2 点坐标
corners = approxPolyDP(hull, 4);  % 强制拟合四边形

% 可视化角点
imshow(bw); hold on;
plot(corners([1:end 1],1), corners([1:end 1],2), 'r-', 'LineWidth', 2);
title('Detected Quadrilateral');

% === Step 3: 设置目标矩形大小（单位：像素） ===
width = max(norm(corners(1,:) - corners(2,:)), norm(corners(3,:) - corners(4,:)));
height = max(norm(corners(1,:) - corners(4,:)), norm(corners(2,:) - corners(3,:)));
width = round(width);
height = round(height);

% === Step 4: 透视变换 ===
src_pts = double(corners);
dst_pts = [0 0; width-1 0; width-1 height-1; 0 height-1];

% 获取 projective 变换矩阵
tform = fitgeotrans(src_pts, dst_pts, 'projective');

% 应用变换
outputRef = imref2d([height, width]);
warped = imwarp(img, tform, 'OutputView', outputRef);

figure; imshow(warped);
title('Warped (Rectified) Output');



function approx = approxPolyDP(points, n)
    % 使用 PCA + 几何中心 + kmeans 简化为 n 边形（n=4 表示四边形）
    % 输入: points 是 N×2 的轮廓点，n 是期望的角点数
    % 输出: approx 是 n×2 的近似多边形顶点

    % 聚类成 n 类（默认按角度散布）
    [~, C] = kmeans(points, n, 'MaxIter', 500, 'Replicates', 10);
    
    % 按角度排序
    centroid = mean(C);
    angles = atan2(C(:,2) - centroid(2), C(:,1) - centroid(1));
    [~, idx] = sort(angles);
    approx = C(idx, :);
end
