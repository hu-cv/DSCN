
function [feat F] = getFeatureRep(I,nbin)
global config
if size(I,3) == 3 && config.use_color
    I = uint8(255*RGB2Lab(I));
elseif size(I,3) == 3
    I = rgb2gray(I);
end
fd = config.fd;
ksize = (1/config.ratio)*4;
if mod(ksize,2) == 0
    ksize = ksize + 1;
end
if config.use_iif
    F{1} = 255-calcIIF(I(:,:,1),[ksize ksize],nbin);%IIF2(I(:,:,1)*255, hist_mtx1, nbin);%feature by pixel ordering
else
    F{1} = uint8(zeros([size(I,1),size(I,2)]));
end
F{2} = I(:,:,1);
if config.use_color
    F{3} = I(:,:,2);
    F{4} = I(:,:,3);
end
if config.use_raw_feat
    feat = double(reshape(cell2mat(F),size(F{1},1),size(F{1},2),[]))/255;    
else
    if ~config.use_color
        feat = zeros([size(I(:,:,1)),2*config.fd]);
    else
        feat = zeros([size(I(:,:,1)),4*config.fd]);
    end
    for i = 1:numel(F)
        feat(:,:,(i-1)*fd+1:i*fd) = bsxfun(@gt,repmat(F{i},[1 1 fd]), reshape(config.thr,1,1,[]));
    end    
end
F = double(reshape(cell2mat(F),size(F{1},1),size(F{1},2),[]));


