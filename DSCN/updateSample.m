function updateSample(I_vf,I,sample_sz,radius,rescale)
global sampler;
global svm_tracker;
global experts;
global config;
roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2);
refer_win = svm_tracker.output;
refer_win(1:2) = refer_win(1:2)/rescale+0.5*(1/rescale-1)*refer_win(3:4)-roi_reg(1:2)+1;%0.5*roi_reg(3:4) - 0.5*refer_win(3:4);
r = radius*(norm([sampler.template_width,sampler.template_height]))/2;
step = max(floor(2*r/sqrt(sample_sz)),1);
x_sample = -r:step:r;
y_sample = -r:step:r;
[X Y] = meshgrid(x_sample,y_sample);
temp = repmat(refer_win,[numel(X),1]);
temp(:,1:2) = temp(:,1:2) + [X(:),Y(:)];
valid_sample = ~(temp(:,1)<1 | temp(:,2)<1 | temp(:,1)+temp(:,3)-1>size(I_vf,2) | temp(:,2)+temp(:,4)-1>size(I_vf,1));
temp = temp(valid_sample,:);
state_dt = temp;
state_dt(:,1) = state_dt(:,1)+sampler.roi(1)-1;
state_dt(:,2) = state_dt(:,2)+sampler.roi(2)-1;
confidence = zeros(size(state_dt,1),1);
confidence_exp = zeros(size(state_dt,1),1);
expert_w = experts{end}.w;% dirty here, need to clean up: expert is the last one
expert_b = experts{end}.Bias;
feat_tmp = zeros(sampler.template_height,sampler.template_width,size(I_vf,3)*size(state_dt,1));
for i=1:size(state_dt,1)
    rect = temp(i,:);
    ch = (i-1)*size(I_vf,3)+1;
    feat_tmp(:,:,ch:ch+size(I_vf,3)-1) = (I_vf(round(rect(2): (rect(2)+rect(4)-1)),round(rect(1):(rect(1)+rect(3)-1)),:));
end
feat_tmp = imresize(feat_tmp,config.template_sz,'nearest');
feat_tmp = reshape(feat_tmp,round(prod(config.template_sz(1:2))*size(I_vf,3)),[]);
confidence = -(svm_tracker.w*feat_tmp + svm_tracker.Bias);
[Y Id] = max(confidence);
svm_tracker.output = state_dt(Id,:);
svm_tracker.confidence = confidence(Id);% 
sampler.patterns_dt = [feat_tmp(:,Id)';sampler.patterns_dt];
sampler.state_dt = [state_dt(Id,:);sampler.state_dt];
sampler.costs = 1-getIOU(sampler.state_dt,svm_tracker.output);
end

