function  distance_cos = mahal_theta_kfold_basis_b_ct(data,theta,n_folds)
%%
theta=circ_dist(theta,0); % to make makre that orientations are centered around 0

u_theta=unique(theta);

train_partitions = cvpartition(theta,'KFold',n_folds); % split data n times using Kfold
distances=nan(length(u_theta),size(data,1),size(data,3),size(data,3)); % prepare for output 

theta_dist=circ_dist2(u_theta',theta)';
%%
for tst=1:n_folds % run for each fold
    trn_ind = training(train_partitions,tst); % get training trial rows
    tst_ind = test(train_partitions,tst); % get test trial rows
    trn_dat = data(trn_ind,:,:); % isolate training data
    tst_dat = data(tst_ind,:,:); % isolate test data
    trn_theta =theta(trn_ind);
    m=double(nan(length(u_theta),size(data,2),size(data,3)));
    m_temp=double(nan(length(u_theta),size(data,2),size(data,3)));
    n_conds_min = min(histc(trn_theta,u_theta)); % get number of trials in each condition
    % average trials over each condition of training data
    % subsample to equalize trial numbers of all bins
    for c=1:length(u_theta)
        temp1=trn_dat(trn_theta==u_theta(c),:,:);
        ind=randsample(1:size(temp1,1),n_conds_min);
        m_temp(c,:,:)=mean(temp1(ind,:,:),1);
    end
     cosfun    = @(theta,mu)((0.5 + 0.5.*cos((theta-mu))).^(length(u_theta)-1));
    for c=1:length(unique(u_theta))
        m(c,:,:)=sum(bsxfun(@times,m_temp(:,:,:),cosfun(u_theta,u_theta(c))))./sum(cosfun(u_theta,u_theta(c)));
    end
    tst_dat_re=reshape(permute(tst_dat,[1 3 2]),[size(tst_dat,1)*size(tst_dat,3),size(tst_dat,2)]);
    for t=1:size(data,3) % decode at each time-point
        if ~isnan(trn_dat(:,:,t))
            % compute pair-wise mahalabonis distance between test-trials
            % and averaged training data, using the covariance matrix
            % computed from the training data
            temp=pdist2(squeeze(m(:,:,t)), squeeze(tst_dat_re),'mahalanobis',covdiag(trn_dat(:,:,t)));
            distances(:,tst_ind,:,t)=reshape(temp,[size(temp,1),size(tst_dat,1),size(tst_dat,3)]);
            % linear regression for each trial
        end
    end
end
distance_cos=squeeze(-mean(bsxfun(@times,cos(theta_dist)',distances),1)); % take cosine-weigthed mean of distances


