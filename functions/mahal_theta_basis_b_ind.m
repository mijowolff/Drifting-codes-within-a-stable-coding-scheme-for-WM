function  [distance_cos,distances] = mahal_theta_basis_b_ind(data_test,theta_test,data_train,theta_train)
% orientation reconstruction, training on independent data, balanced
% training set through subsampling
%% input

% data, format is trial by channel by time, test data

% theta is vector with angles in radians for each trial (must comprise the
% whole circle, thus, for orientation data, which is only 180 degrees, make
% sure to multiply by 2). This function assumes a finite number of unique
% angles


%% output
% output is trial by time, to summarize average over trials

% distance_cos  is a measure of decoding accuracy, cosine weighted distances
% of pattern-difference between trials of increasinglt dissimilar
% orientations

% distances is the ordered mean-centred distances

%%
theta_test=circ_dist(theta_test,0);
theta_train=circ_dist(theta_train,0);

u_theta=unique(theta_test);

distances=nan(length(u_theta),size(data_test,1),size(data_test,3)); % prepare for output

theta_dist=circ_dist2(u_theta',theta_test)';

trn_dat = data_train; % isolate training data
tst_dat = data_test; % isolate test data

m=double(nan(length(u_theta),size(data_train,2),size(data_train,3)));
m_temp=double(nan(length(u_theta),size(data_train,2),size(data_train,3)));
n_conds = [u_theta,histc(theta_train,u_theta)];% get number of trials in each condition
% random subsampling to equalize number of trials of each condition
for c=1:length(u_theta)
    temp1=trn_dat(theta_train==u_theta(c),:,:);
    ind=randsample(1:size(temp1,1),min(n_conds(:,2)));
    m_temp(c,:,:)=mean(temp1(ind,:,:),1);
end
% smooth over orientations in the training data using a predetermined
% basis set
cosfun    = @(theta,mu)((0.5 + 0.5.*cos((theta-mu))).^(length(u_theta)-1));
for c=1:length(u_theta)
    m(c,:,:)=sum(bsxfun(@times,m_temp(:,:,:),cosfun(u_theta,u_theta(c))))./sum(cosfun(u_theta,u_theta(c)));
end
for t=1:size(data_test,3) % decode at each time-point
    if ~isnan(trn_dat(:,:,t))
        % compute pair-wise mahalabonis distance between test-trials
        % and averaged training data, using the covariance matrix
        % computed from the training data
        temp=pdist2(squeeze(m(:,:,t)), squeeze(tst_dat(:,:,t)),'mahalanobis',covdiag(trn_dat(:,:,t)));
        distances(:,:,t)=temp;
    end
end
distance_cos=-mean(bsxfun(@times,cos(theta_dist)',distances),1);% take cosine-weigthed mean of distances
% reorder distances so that same condition distance is in the middle
for c=1:length(u_theta)
    temp=round(circ_dist(u_theta,u_theta(c)),4);
    temp(temp==round(pi,4))=round(-pi,4);
    [~,i]=sort(temp);
    distances(:,theta_test==u_theta(c),:)=distances(i,theta_test==u_theta(c),:);
end
distances=-bsxfun(@minus,distances,mean(distances,1)); % mean-centre distances