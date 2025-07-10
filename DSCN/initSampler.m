function initSampler(init_rect,I_vf,I,use_color)
global config
global sampler;
init_rect_roi = init_rect;
init_rect_roi(1:2) = init_rect(1:2) - sampler.roi(1:2)+1;
template = I_vf (round(init_rect_roi(2):init_rect_roi(2)+init_rect_roi(4)-1),...
    round(init_rect_roi(1):init_rect_roi(1)+init_rect_roi(3)-1),:);
sampler.template = imresize(template,config.template_sz);
sampler.template_size = size(sampler.template);
sampler.template = sampler.template(:)';
sampler.template_width = init_rect(3);
sampler.template_height = init_rect(4);
if use_color
    sampler.feature_num = 4;
else
    sampler.feature_num = 2;
end
resample(I_vf); 




