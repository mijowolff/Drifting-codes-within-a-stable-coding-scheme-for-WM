function  [dec_acc,distance_difference,pred_cond,distances] = mahal_nominal_kfold_b(data,conditions,n_folds)
% nominal/categorcial decoding with balancing of training data through
% subsampling

%% input
% "data" is trials by channels by time
% classifier runs at each time-point (third dimension of data) separately
% "conditions" is a vector containing the conditions labels
% "n_folds" is the number of folds for training and testing

%% output
% dec_acc are "hits" (1) and "misses" (0), for correctly, and incorectly
% classified trials

% "distance difference" is the difference in distance between same and
% different conditions. Can be used as the decoding performance

% "pred_cond" are the predicted conditions by the classifer. 

% "distances" are all the distances between conditions

%%
train_partitions = cvpartition(conditions,'KFold',n_folds); % split data n times using Kfold
distances=nan(length(unique(conditions)),size(data,1),size(data,3)); % prepare for output
distance_difference=nan(size(data,1),size(data,3));
conds_u=unique(conditions);
for fold=1:n_folds % run for each fold
    trn_ind = training(train_partitions,fold); % get training trial rows
    tst_ind = test(train_partitions,fold); % get test trial rows
    trn_dat = data(trn_ind,:,:); % isolate training data
    tst_dat = data(tst_ind,:,:); % isolate test data
    trn_cond =conditions(trn_ind);    
    m=(nan(length(unique(conditions)),size(data,2),size(data,3)));
    n_conds=(histc(trn_cond,conds_u)); % get number of trials in each condition
    
    % random subsampling to equalize number of trials of each condition
    for c=1:length(unique(conds_u))
        temp1=trn_dat(trn_cond==conds_u(c),:,:);
        m(c,:,:)=mean(temp1(randsample(1:n_conds(c),min(n_conds)),:,:),1);
    end
    for t=1:size(data,3) % decode at each time-point
            % compute pair-wise mahalabonis distance between test-trials
            % and averaged training data, using the covariance matrix
            % computed from the training data
            distances(:,tst_ind,t)=pdist2(squeeze(m(:,:,t)), squeeze(tst_dat(:,:,t)),'mahalanobis',covdiag(trn_dat(:,:,t)));
    end
end
[~,pred_cond]=(min(distances,[],1));
pred_cond=squeeze(pred_cond);
dec_acc=sum(pred_cond==conditions')./length(conditions);
% compute distance difference between same-condition distance and
% different-condition difference
for c=1:length(unique(conditions))
    distance_difference(conditions==c,:)=  mean(distances(setdiff(unique(conditions),c),conditions==c,:),1)-distances(c,conditions==c,:);
end


