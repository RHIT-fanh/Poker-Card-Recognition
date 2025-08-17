% 图像反转批处理程序 - Chengyang Ye 使用版本

% 清除环境
clc; clear; close all;

% 设置图像文件夹路径（当前文件夹）
img_folder = '.';  % 或替换为绝对路径，例如 'D:\Images'

% 获取所有 PNG 文件
png_files = dir(fullfile(img_folder, '*.png'));

% 检查是否有文件
if isempty(png_files)
    disp('未找到任何 PNG 图像文件。');
    return;
end

% 批量处理
for k = 1:length(png_files)
    % Step 1: 构造完整路径并读取图像
    filename = png_files(k).name;
    filepath = fullfile(img_folder, filename);
    img = imread(filepath);

    % Step 2: 反转像素值
    inverted_img = 255 - img;

    % Step 3: 显示原图和反转图（可注释）
    figure;
    subplot(1,2,1); imshow(img); title(['原始图像 - ' filename], 'Interpreter', 'none');
    subplot(1,2,2); imshow(inverted_img); title('反转图像');

    % Step 4: 保存反转图像
    [~, base, ~] = fileparts(filename);  % 提取不带扩展名的文件名
    out_name = [base '_flipped.png'];
    imwrite(inverted_img, fullfile(img_folder, out_name));
    
    fprintf('已保存反转图像：%s\n', out_name);
end
