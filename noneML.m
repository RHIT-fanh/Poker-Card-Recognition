clear; 
% close all;

% img_rgb = imread('2C10.jpg');
img_rgb = imread('000554394_jpg.rf.8be18f32226fb0d2208dddd4349077ba_04_cls6.jpg');

%% extractROI
gray  = rgb2gray(img_rgb);
blur  = imgaussfilt(gray, 1);
level = graythresh(blur);           % 自动阈值 [0,1]
bin   = ~imbinarize(blur, level);   % 白色为前景

figure; imshow(bin);

stats = regionprops(bin,'BoundingBox','Area','Image');
valid_stats = stats([stats.Area] > 1 & [stats.Area] < 16000);
fprintf('Detected %d candidate ROIs\n', numel(valid_stats));

%% get the templates
tmp_dir  = 'D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\try\template';
tmp_files= dir(fullfile(tmp_dir,'*_hu.mat'));

%% caluculate the distanse with template
labels = cell(1,numel(valid_stats));
for k = 1:numel(valid_stats)
    roi  = valid_stats(k).Image;   % 提取二值 ROI 图像

    % 显示这个 ROI 的二值图
    figure; imshow(roi); title(sprintf('ROI %d - Binary Region', k));

    hu   = computeHuMoments(roi);
    logH = -sign(hu).*log10(abs(hu)+eps);
    fprintf('ROI %d  Hu:\n'); disp(hu);

    best_d = inf; best='';
    for f = 1:numel(tmp_files)
        d  = load(fullfile(tmp_dir,tmp_files(f).name));
        vv = -sign(d.hu).*log10(abs(d.hu)+eps);
        dist = norm(logH-vv);
        if dist<best_d, best_d=dist; best=erase(tmp_files(f).name,'_hu.mat'); end
    end
    labels{k}=best;
    fprintf('ROI %d → %s (%.4f)\n\n', k, best, best_d);
end



 % show the bouding box and classification result
 figure;
imshow(img_rgb); title('Recognition Result'); hold on;
for k = 1:numel(valid_stats) 
    rectangle('Position',valid_stats(k).BoundingBox,'EdgeColor','r','LineWidth',2);
    text(valid_stats(k).BoundingBox(1),valid_stats(k).BoundingBox(2)-15,labels{k},...
        'Color','cyan','FontSize',12,'FontWeight','bold');
end

imshow(img_rgb); title('Recognition Result'); hold on;

for k = 1:numel(valid_stats)
    bbox = valid_stats(k).BoundingBox;
    
    
    rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
    
    
    text(bbox(1), bbox(2) - 15, labels{k}, ...
        'Color','cyan', 'FontSize',12, 'FontWeight','bold');
    
    
    text(bbox(1), bbox(2) + bbox(4) + 5, sprintf('#%d', k), ...
        'Color','yellow', 'FontSize',10, 'FontWeight','bold');
end


