function roi = get_roi_mask(I)
% input: RGB or gray image
% output: ROI region in binary

gray  = rgb2gray(I);               
blur  = imgaussfilt(gray, 1);       
bin   = imbinarize(blur, 150/255);  % normalization for threshold
clean = bwareaopen(bin, 300);       % clean small regions

stats = regionprops(clean, 'Area', 'Image');
assert(~isempty(stats), 'cannot find foreground！');

[~, idx] = max([stats.Area]);      
roi = stats(idx).Image;            
end
