function resample(I_vf,step_size)
global sampler;
global CNN_tracker;
global config;
if nargin < 2
    step_size = max(round(sampler.template_size(1:2)/5),1);
end
feature_map = imresize(I_vf,config.ratio,'nearest');
step_size = max(round(min(sampler.template_size(1:2))/4),1);
step_size = step_size([1 1]);
rect=CNN_tracker.output;
upleft = round([rect(1)-sampler.roi(1)+1,rect(2)-sampler.roi(2)+1]);
if ~((upleft(1)<1) || (upleft(2)<1) || (round(upleft(1)+rect(3)-1)>size(I_vf,2)) || (round(upleft(2)+rect(4)-1)>size(I_vf,1)))
    sub_win=I_vf(round(upleft(2): (upleft(2)+rect(4)-1)),round(upleft(1): (upleft(1)+rect(3)-1)),:);
    output_feat = imresize(sub_win,config.template_sz);
    CNN_tracker.output_feat = output_feat(:)';
else    
    warning('tracking window outside of frame');
    keyboard
end
feature_map=double(feature_map);
if size(feature_map,3)>1
sampler.patterns_dt = [im2colstep(feature_map,sampler.template_size,[step_size, size(I_vf,3)])';...
    CNN_tracker.output_feat];
else
    
 sampler.patterns_dt = [im2colstep(feature_map,sampler.template_size,step_size)';...
    CNN_tracker.output_feat];  
end
temp = repmat(rect,[size(sampler.patterns_dt,1),1]);
 N=size(sampler.patterns_dt,1);
% sampler.template_size
% size(sampler.patterns_dt)
sampler.patterns_dt=reshape(sampler.patterns_dt',[sampler.template_size,N]);
S=size(sampler.patterns_dt);
if numel(S)<4
temp1=zeros(S(1),S(2),1,S(3));
temp1(:,:,1,:)=sampler.patterns_dt;
sampler.patterns_dt=temp1;
end
[X Y] = meshgrid(1:step_size(2):size(feature_map,2)-sampler.template_size(2)+1,1:step_size(1):size(feature_map,1)-sampler.template_size(1)+1);
temp(1:end-1,1) = (X(:)-1)/config.ratio + sampler.roi(1);
temp(1:end-1,2) = (Y(:)-1)/config.ratio + sampler.roi(2);
left = max(round(temp(:,1)),round(rect(1)));
top = max(round(temp(:,2)),round(rect(2)));
right = min(round(temp(:,1)+temp(:,3)),round(rect(1)+rect(3)));
bottom = min(round(temp(:,2)+temp(:,4)),round(rect(2)+rect(4)));
ovlp = max(right - left,0).*max(bottom - top, 0);
sampler.costs = 1 - ovlp./(2*rect(3)*rect(4)-ovlp);
sampler.state_dt = temp;
end

