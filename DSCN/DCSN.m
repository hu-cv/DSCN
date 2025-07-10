
function result = DCSN(input, ext, show_img, init_rect, sframe, end_frame)
addpath(genpath('.'));
D = dir(fullfile(input,['*.', ext]));
file_list={D.name};
if nargin < 4
    init_rect = -ones(1,4);
end
if nargin < 5
    sframe = 1;
end
if nargin < 6
    end_frame = numel(file_list);
end
global sampler
global CNN_tracker
global experts
global config
global finish
config.display = true;
sampler = createSampler();
experts = {};
finish = 0;
timer = 0;
result.res = nan(end_frame-sframe+1,4);
result.len = end_frame-sframe+1;
result.startFrame = sframe;
result.type = 'rect';
para.opt = struct('numsample', 1, 'affsig', [0,0,0,0,0,0]);
para.normalWidth = 240;
para.normalHeight = 120;
para.pars.K        = [];                    
para.pars.max_iter = 50;	 
para.pars.lambda   = 0.1;  
para.pars.gamma    = 0.1;  
if show_img
    figure(1); set(1,'KeyPressFcn', @handleKey);
end
output = zeros(1,4);
patchsize1 = [5 5];
patchsize2 = [3 3];
patchnum(1) = length(patchsize1(1)/2 : 1: (sz(1)-patchsize1(1)/2));
patchnum(2) = length(patchsize1(2)/2 : 1: (sz(2)-patchsize1(2)/2));
patch2num(1) = length(patchsize2(1)/2 : 1: (sz(1)-patchsize2(1)/2));
patch2num(2) = length(patchsize2(2)/2 : 1: (sz(2)-patchsize2(2)/2));
Fisize1 = 64;
Fisize2 = 96;
Fisize3 = 128;
for frame_id = sframe:end_frame
    if finish == 1
        break;
    end
    if ~config.display
        clc
        display(input);
        display(['frame: ',num2str(frame_id),'/',num2str(end_frame)]);
    end
    I_orig=imread(fullfile(input,file_list{frame_id}));
    if frame_id==sframe
        if isequal(init_rect,-ones(1,4))
            assert(config.display)
            figure(1)
            imshow(I_orig);
            [InitPatch init_rect]=imcrop(I_orig);
        end
        init_rect = round(init_rect);
        config = makeConfig(I_orig,init_rect,true,true,true,show_img);
        CNN_tracker.output = init_rect*config.image_scale;
        CNN_tracker.output(1:2) = CNN_tracker.output(1:2) + config.padding;
        CNN_tracker.output_exp = CNN_tracker.output;
        output = CNN_tracker.output;
    end
    [I_scale]= getFrame2Compute(I_orig);
    if frame_id == sframe
        sampler.roi = rsz_rt(CNN_tracker.output,size(I_scale),5*config.search_roi,false);
    else
        sampler.roi = rsz_rt(output,size(I_scale),config.search_roi,true);
    end
    I_crop = I_scale(round(sampler.roi(2):sampler.roi(4)),round(sampler.roi(1):sampler.roi(3)),:);
    tic
    if frame_id==sframe
        para.pars.K = Fisizeones(1,2);
        neg1 = sampleNeg(img, param.est', opt.psize, 20, opt, 8);
        FiNeg1 = zeros(64,patchnum(1)*patchnum(2));
        for i = 1:size(neg1,2)
            FiNeg1 = FiNeg1 + affineTrainNeg(reshape(neg1(:,i),[48 48]), patchsize1, patchnum, Fisize1);
        end
        FiNeg1 = FiNeg1/size(FiNeg1,2);
        image = warpimg(I_orig, param0, opt.psize);
        neg2 = sampleNeg(image, param.est', opt.psize, 20, opt, 8);
        [Fio1, patcho] = designDSD(image, patchsize1, patchnum2, Fisize1, FiNeg1, para);
        image2 = Fio1'*image;
        FiNeg2 = zeros(96,patchnum2(1)*patchnum2(2));
        for i = 1:size(neg,2)
            FiNeg2 = FiNeg1 + affineTrainNeg(reshape(neg2(:,i),[48 48]), patchsize2, patch2num, Fisize2);
        end
        FiNeg2 = FiNeg2/size(neg,2);
        [Fio2, patcho]= designDSD(image2, patchsize2, patchnum, Fisize2, FiNeg2, para);
        image3 = Fio2'*image2;
        neg3 = sampleNeg(FiNeg2, param.est', opt.psize, 20, opt, 8);
        FiNeg3 = zeros(128,patchnum2(1)*patchnum2(2));
        
        for i = 1:size(neg,2)
            FiNeg3 = FiNeg3 + affineTrainNeg(reshape(neg3(:,i),[48 48]), patchsize3, patch2num, Fisize3);
        end
        FiNeg3 = FiNeg3/size(neg,2);
        [Fio3, patcho]= designDSD(image3, patchsize3, patchnum, Fisize3, FiNeg3, para);
        initSampler(CNN_tracker.output,I_crop,[],config.use_color);
        train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
        label = sampler.costs(train_mask,1)<config.thresh_p;
        fuzzy_weight = Fio3;
        createCNNTracker
        global CNN_tracker;
        global experts;
        sample_w = fuzzy_weight;
        pos_mask = label>0.5;
        neg_mask = ~pos_mask;
        s1 = sum(sample_w(pos_mask));
        s2 = sum(sample_w(neg_mask));
        lambda=CNN_tracker.lambda;
        N=round(1+2*s2/s1);
        S=size(sample);
        pos=sample(:,:,:,pos_mask);
        pos=repmat(pos,1,1,1,N);
        pos_label=label(pos_mask);
        pos=double(pos)+double(0.05*randn(size(pos)));
        pos_label=repmat(pos_label,N,1);
        sample=cat(numel(S),sample,pos);
        label=[label;pos_label];
        sample=single(sample);
        H=net.eval({'input', sample});
        H=H(end).x;
        size_temp=size(H);
        H=reshape(H,[],size_temp(4))';
        sample1=reshape(sample,[],size_temp(4))';
        sample1=normalize_feature(sample1);
        H=[H,sample1];
        if size(H,1)>size(H,2)
            beta=(H'*H+lambda*eye(size(H,2)))\H'*label;
        else
            beta=H'*((lambda*eye(size(H,1))+H*H')\label);
        end
        CNN_tracker.beta=beta;
        M=(H'*H+lambda*eye(size(H,2)))\eye(size(H,2));
        CNN_tracker.M=M;
        experts{1}.score = [];
        experts{1}.snapshot = CNN_tracker;
        experts{2} = experts{1};
        if config.display
            figure(1);
            imshow(I_orig);
            res = CNN_tracker.output;
            res(1:2) = res(1:2) - config.padding;
            res = res/config.image_scale;
            rectangle('position',res,'LineWidth',2,'EdgeColor','b')
        end
    else
        if config.display
            figure(1)
            imshow(I_orig);
            roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2)+1;
            roi_reg(1:2) = roi_reg(1:2) - config.padding;
            rectangle('position',roi_reg/config.image_scale,'LineWidth',1,'EdgeColor','r');
        end
        if mod((frame_id - sframe + 1),config.expert_update_interval) == 0% CNN_tracker.update_count >= config.update_count_thresh
            updateTrackerExperts;
        end
        expertsDo(I_crop,config.expert_lambda,config.label_prior_sigma);
        output = CNN_tracker.output;
        if config.display
            figure(1)
            res = output;
            res(1:2) = res(1:2) - config.padding;
            res = res/config.image_scale;
            if CNN_tracker.best_expert_idx ~= numel(experts)
                res_prev = CNN_tracker.output_exp;
                res_prev(1:2) = res_prev(1:2) - config.padding;
                res_prev = res_prev/config.image_scale;
                rectangle('position',res_prev,'LineWidth',2,'EdgeColor','r')
                rectangle('position',res,'LineWidth',2,'EdgeColor','y')
            else
                rectangle('position',res,'LineWidth',2,'EdgeColor','b')
            end
        end
        
        CNN_tracker.update_count=0;
        if CNN_tracker.confidence <0.9
            % disp('updating')
            train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
            label = sampler.costs(train_mask) < config.thresh_p;
            skip_train = false;
            if ~skip_train
                 negX = sampleNeg(img, param.est', opt.psize, 20, opt, 8);
                 FiNegX = zeros(36,Fisize2);
                 for i = 1:size(negX,2)
                     FiNegX = FiNegX + affineTrainNeg(reshape(negX(:,i),[48 48]), patchsize2, patch2num, Fisize3);
                 end
                 FiNegX = FiNegX/size(negX,2);
                [FioX, patcho] = designFilters(I_crop, param.est', opt, patchsize2, patch2num, Fisize3, FiNegX, para);
                costs = sampler.costs(train_mask);
                fuzzy_weight = FioX;
                fuzzy_weight(~label) = 2*costs(~label)-1;
                updateCNNTracker (sampler.patterns_dt(:,:,:,train_mask),label,fuzzy_weight);
                CNN_tracker.update_count=0;
            end
        else
            CNN_tracker.update_count = CNN_tracker.update_count+1;
        end
    end
    timer = timer + toc;
    res = output;
    res(1:2) = res(1:2) - config.padding;
    result.res(frame_id-sframe+1,:) = res/config.image_scale;
end
result.fps = result.len/timer;
clearvars -global sampler CNN_tracker experts config finish
