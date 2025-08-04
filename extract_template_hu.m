clear; 

img_path = 'test.tif';                 
[~, base, ~] = fileparts(img_path); 

I   = imread(img_path);
roi = get_roi_mask(I);              

hu  = computeHuMoments(roi);

save([base '_hu.mat'], 'hu');       
disp(['saved: ' base '_hu.mat']);
disp('Hu moments:');  disp(hu);

