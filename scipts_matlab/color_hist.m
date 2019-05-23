im = imread('dark_crop.jpg');

im = rgb2hsv(im);

Red = im(:,:,1);
Green = im(:,:,2);
Blue = im(:,:,3);

[yRed, x] = imhist(Red);
[yGreen, x] = imhist(Green);
[yBlue, x] = imhist(Blue);

yRed(1) = 0;
yGreen(1) = 0;
yBlue(1) = 0;

% figure;
% plot(x, yRed, 'r', x, yGreen, 'Green', x, yBlue, 'Blue');

im2 = imread('state1_crop.jpg');
im2 = rgb2hsv(im2);

Red2 = im2(:,:,1);
Green2 = im2(:,:,2);
Blue2 = im2(:,:,3);

[yRed2, x] = imhist(Red2);
[yGreen2, x] = imhist(Green2);
[yBlue2, x] = imhist(Blue2);

yRed2(1) = 0;
yGreen2(1) = 0;
yBlue2(1) = 0;

im2 = imread('state0_crop.jpg');
im2 = rgb2hsv(im2);

Red2 = im2(:,:,1);
Green2 = im2(:,:,2);
Blue2 = im2(:,:,3);

[yRed3, x] = imhist(Red2);
[yGreen3, x] = imhist(Green2);
[yBlue3, x] = imhist(Blue2);

yRed3(1) = 0;
yGreen3(1) = 0;
yBlue3(1) = 0;



figure;
plot(x, yRed, 'r--', x, yGreen, 'g--', x, yBlue, 'b--',x, yRed2, 'r-', x, yGreen2, 'g-', x, yBlue2, 'b-',x, yRed3, 'r:', x, yGreen3, 'g:', x, yBlue3, 'b:');

