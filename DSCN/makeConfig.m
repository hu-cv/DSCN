function config = makeConfig(frame,selected_rect,use_color,use_experts,use_iif, show_img)
warning('off','MATLAB:maxNumCompThreads:Deprecated');
maxNumCompThreads(1);
if nargin < 5
    use_iif = true;
end
config.search_roi = 2; 
config.padding = 40; 
config.debug = false;
config.verbose = false;
config.display = show_img; 
config.use_experts = use_experts;
config.use_raw_feat = false; 
config.use_iif = use_iif;
config.max_expert_sz = 4; 
config.expert_update_interval = 30;
config.update_count_thresh = 1;
config.entropy_score_winsize = 4;
config.expert_lambda = 100;
config.label_prior_sigma = 15;
config.hist_nbin = 32;
config.thresh_p = 0.1; 
config.thresh_n = 0.5; 
config.use_color = false;
if (size(frame,3) == 3 && ~isequal(frame(:,:,1),frame(:,:,2),frame(:,:,3))) && use_color
    config.use_color = true;    
end
if config.use_color
    thr_n = 5; 
else
    thr_n = 9;
end
config.thr = (1/thr_n:1/thr_n:1-1/thr_n)*255;
config.fd = numel(config.thr);
frame_min_width = 320;
trackwin_max_dimension = 64;
template_max_numel = 144;
frame_sz = size(frame);
if max(selected_rect(3:4)) <= trackwin_max_dimension ||...
        frame_sz(2) <= frame_min_width
    config.image_scale = 1;
else
    min_scale = frame_min_width/frame_sz(2);
    config.image_scale = max(trackwin_max_dimension/max(selected_rect(3:4)),min_scale);    
end
wh_rescale = selected_rect(3:4)*config.image_scale;
win_area = prod(wh_rescale);
config.ratio = (sqrt(template_max_numel/win_area));
template_sz = round(wh_rescale*config.ratio); 
config.template_sz = template_sz([2 1]);

