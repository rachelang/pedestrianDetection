function img = bmpToMatrix(fileName)

rgb = imread(fileName);
dlmwrite('testImage.txt', rgb);
img = dlmread('testImage.txt');

end