function updateTrackerExperts
global config
global CNN_tracker
global experts
if numel(experts) < config.max_expert_sz
    CNN_tracker.update_count = 0;
    %experts{end}.snapshot = CNN_tracker;
    experts{end+1} = experts{end};
else
     experts(1:end-1) = experts(2:end);
     experts{end}.snapshot = CNN_tracker;
end