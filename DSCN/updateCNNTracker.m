function updateCNNTracker(sample,label,sample_w)
global config;
global CNN_tracker;
global experts;
Block=size(sample,1);
pos_mask = label>0.5;
neg_mask = ~pos_mask;
 s1 = sum(sample_w(pos_mask));
 s2 = sum(sample_w(neg_mask));
 N=round(1+2*s2/s1);  
 pos=sample(:,:,:,pos_mask);
pos_label=label(pos_mask);
 pos=repmat(pos,1,1,1,N);
 pos=pos+0.05*randn(size(pos));
 pos_label=repmat(pos_label,N,1);
 sample=cat(4,sample,pos);
label=[label;pos_label]; 
M=CNN_tracker.M;
beta=CNN_tracker.beta;
sample=single(sample);
 H=net.eval({'input', sample});
H=H(end).x;
size_temp=size(H);
H=reshape(H,[],size_temp(4))';
sample1=reshape(sample,[],size_temp(4))';
sample1=normalize_feature(sample1);
H=[H,sample1];
Block=size(H,1);
M=M-M*H'*((eye(Block)+H*M*H')\H*M);
beta=beta+M*H'*(label-H*beta);
CNN_tracker.M=M;
CNN_tracker.beta=beta;
experts{end}.snapshot=CNN_tracker;      


