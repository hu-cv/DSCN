function [Fi, patch] = designDSD(image, patchsize, patchnum, Fisize, FiNeg, para)
patch = zeros(prod(patchsize), prod(patchnum));
blocksize = size(image);
y = patchsize(1)/2;
x = patchsize(2)/2;
patch_centy = y : 1: (blocksize(1)-y);
patch_centx = x : 1: (blocksize(2)-x);
l =1;
for j = 1: patchnum(1)                 
    for k = 1:patchnum(2)
        data = image(patch_centy(j)-y+1 : patch_centy(j)+y, patch_centx(k)-x+1 : patch_centx(k)+x);
        patch(:, l) = reshape(data,numel(data),1);
        l = l+1;
    end
end
Y = normVector([double(patch),FiNeg]);
pars.K = Fisize;
Fi = DSD(Y, para.pars);
