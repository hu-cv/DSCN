function [out,a,b] = whitening(in)
MN = size(in,1);
a = mean(in);
b = std(in)+1e-14;
out = (in - ones(MN,1)*a) ./ (ones(MN,1)*b);