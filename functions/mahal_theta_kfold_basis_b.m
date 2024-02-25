function  [distance_cos,distances] = mahal_theta_kfold_basis_b(data,theta,n_folds,data_trn)
% orientation reconstruction using cross-validation, balanced
% training set through subsampling
%% input

% data format is trial by channel by time

% theta is vector with angles in radians for each trial (must comprise the
% whole circle, thus, for orientation data, which is only 180 degrees, make
% sure to multiply by 2). This function assumes a finite number of unique
% angles

% "n_folds" is the number of folds for training and testing

%% output
% output is trial by time, to summarize average over trials

% distance_cos  is a measure of decoding accuracy, cosine weighted distances
% of pattern-difference between trials of increasinglt dissimilar
% orientations

% distances is the ordered mean-centred distances
%%
theta=circ_dist(theta,0); % to make makre that orientations are centered around 0

u_theta=unique(theta);

train_partitions = cvpartition(theta,'KFold',n_folds); % split data n times using Kfold
distances=nan(length(u_theta),size(data,1),size(data,3)); % prepare for output 

theta_dist=circ_dist2(u_theta',theta)';

% if no optional training data from different period in trial is provided,
% then test and training data are the same
if nargin==3
    data_trn=data;
end

for tst=1:n_folds % run for each fold
    trn_ind = training(train_partitions,tst); % get training trial rows
    tst_ind = test(train_partitions,tst); % get test trial rows
    trn_dat = data_trn(trn_ind,:,:); % isolate training data
    tst_dat = data(tst_ind,:,:); % isolate test data
    trn_theta =theta(trn_ind);
    m=double(nan(length(u_theta),size(data,2),size(data,3)));
    n_conds = [u_theta,histc(trn_theta,u_theta)]; % get number of trials in each condition
    m_temp=(nan(length(u_theta),size(data,2),size(data,3)));
    
    % random subsampling to equalize number of trials of each condition
    for c=1:length(u_theta)
        temp1=trn_dat(trn_theta==u_theta(c),:,:);
        ind=randsample(1:size(temp1,1),min(n_conds(:,2)));
        m_temp(c,:,:)=mean(temp1(ind,:,:),1);
    end
    
    % smooth over orientations in the training data using a predetermined
    % basis set
    cosfun    = @(theta,mu)((0.5 + 0.5.*cos((theta-mu))).^(length(u_theta)-1));
    for c=1:length(unique(u_theta))
        m(c,:,:)=sum(bsxfun(@times,m_temp(:,:,:),cosfun(u_theta,u_theta(c))))./sum(cosfun(u_theta,u_theta(c)));
    end
    for t=1:size(data,3) % decode at each time-point
        if ~isnan(trn_dat(:,:,t))
            % compute pair-wise mahalabonis distance between test-trials
            % and averaged training data, using the covariance matrix
            % computed from the training data
            temp=pdist2(squeeze(m(:,:,t)), squeeze(tst_dat(:,:,t)),'mahalanobis',covdiag(trn_dat(:,:,t)));            
            distances(:,tst_ind,t)=temp;
        end
    end
end
distance_cos=-mean(bsxfun(@times,cos(theta_dist)',distances),1); % take cosine-weigthed mean of distances
% reorder distances so that same condition distance is in the middle
for c=1:length(u_theta)
    temp=round(circ_dist(u_theta,u_theta(c)),4);
    temp(temp==round(pi,4))=round(-pi,4);
    [~,i]=sort(temp);
    distances(:,theta==u_theta(c),:)=distances(i,theta==u_theta(c),:);
end
distances=-bsxfun(@minus,distances,mean(distances,1)); % mean-centre distances 
