clear all;
close all;
clc;
%% Figure 5A, from Drifting codes (2020) Wolff et al.
cd('D:\Drifting codes\upload files') %path to main dir.
addpath(genpath(cd))

% channels of interest
test_chans = {'P7';'P5';'P3';'P1';'Pz';'P2';'P4';'P6';'P8';'PO7';'PO3';'POz';'PO4';'PO8';'O2';'O1';'Oz'};

reps=100; % number of repeats for random subsampling
span=5; % number of time-points to average over (5 = 10 ms for 500 Hz)
toi=[.1 .4001];
angspace=-pi+pi/4:pi/2:pi-pi/4; % orientation space
do_MDS=0; %1=run the MDS, or 0= load in previous output
%%
if do_MDS
    for sub=1:26
        fprintf(['Doing ' num2str(sub) '\n'])
        
        load(['dat_' num2str(sub) '.mat']);
        
        % remove trials based on bad performance (>circular SDs)
        report_err=circ_ang2rad(Results(:,6).*2);
        [~,SD]=circ_std(report_err);
        incl_ng=find(abs(circ_dist(report_err,circ_mean(report_err)))<(3*SD));

        % trials to include (based on performance and previously marked
        % "bad" trials
        incl_i=intersect(incl_ng,setdiff(1:size(Results,1), ft_imp1.bad_trials));
        
        % extract good trials, channels and time window of interest of
        % impulse 1 epoch and reformat
        dat_temp=ft_imp1.trial(incl_i,ismember(ft_imp1.label,test_chans),ft_imp1.time>toi(1)&ft_imp1.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3)); % take relative baseline
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard'); % downsample
        dat_temp=dat_temp(:,:,1:span:end);
        dat_imp1=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]); %combine channel and time dimensions
        
        clear ft_imp1
        
        % extract good trials, channels and time window of interest of
        % impulse 2 epoch and reformat
        dat_temp=ft_imp2.trial(incl_i,ismember(ft_imp2.label,test_chans),ft_imp2.time>toi(1)&ft_imp2.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3));
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard');
        dat_temp=dat_temp(:,:,1:span:end);
        dat_imp2=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]);
        
        clear ft_imp2
        
        dat_imps=(dat_imp1+dat_imp2)./2; % combine both impulses
        clear dat_imp1
        clear dat_imp2
        %%
        cue_cond=Results(incl_i,3);
        % always multiply orientaitons (-90 to 90) by 2 to adhere to
        % circular geometry
        cued_rad=Results(incl_i,4).*2;
        
        % replace orientation values to the closes value of the current
        % orientation space
        ang_bin_temp=repmat(angspace,[1 length(cued_rad)]);
        [~,ind]= min(abs(circ_dist2(angspace,cued_rad)));
        bin_cued=ang_bin_temp(ind)';
        
        bin_cued_left=bin_cued(cue_cond==1,1);
        bin_cued_right=bin_cued(cue_cond==2,1);
        
        dat_left=dat_imps(cue_cond==1,:,:);
        dat_right=dat_imps(cue_cond==2,:,:);
        %%
        % compute covariance matrices for mahalanobis distances
        sigma=covdiag(cat(1,dat_right,dat_left));
        
        % get minimum number of trial for subsampling
        out = [angspace',histc(bin_cued_left,angspace')];
        min_cond_left=min(out(:,2));
        out = [angspace',histc(bin_cued_right,angspace')];
        min_cond_right=min(out(:,2));
        min_cond=min(cat(1,min_cond_left,min_cond_right));
        dists_temp=nan(reps,8,8);
        for r=1:reps
            dat_left_av=[]; dat_right_av=[];
            for ang=angspace
                trls=randsample(find(bin_cued_right==ang),min_cond);
                dat_temp=mean(dat_right(trls,:),1);
                dat_right_av=cat(1,dat_right_av,dat_temp);
                trls=randsample(find(bin_cued_left==ang),min_cond);
                dat_temp=mean(dat_left(trls,:),1);
                dat_left_av=cat(1,dat_left_av,dat_temp);
            end
            % get all pairwise distances between orientations and locations
            dists_temp(r,:,:)=squareform(pdist(cat(1,dat_right_av,dat_left_av),'mahalanobis',sigma));
        end
        dists(sub,:,:)=squeeze(mean(dists_temp,1));
    end
else
    load('Fig_5A_results.mat')
end
Coords=cmdscale(squeeze(mean(dists,1)));
%% figure 5A left
figure
scatter3(squeeze(-Coords(1,1)),squeeze(Coords(1,2)),squeeze(Coords(1,3)),'xm','LineWidth',1.5), hold on
scatter3(squeeze(-Coords(2,1)),squeeze(Coords(2,2)),squeeze(Coords(2,3)),'xg','LineWidth',1.5), hold on
scatter3(squeeze(-Coords(3,1)),squeeze(Coords(3,2)),squeeze(Coords(3,3)),'xb','LineWidth',1.5), hold on
scatter3(squeeze(-Coords(4,1)),squeeze(Coords(4,2)),squeeze(Coords(4,3)),'xc','LineWidth',1.5), hold on
scatter3(squeeze(-Coords(5,1)),squeeze(Coords(5,2)),squeeze(Coords(5,3)),'sm','LineWidth',1.5), hold on
scatter3(squeeze(-Coords(6,1)),squeeze(Coords(6,2)),squeeze(Coords(6,3)),'sg','LineWidth',1.5), hold on
scatter3(squeeze(-Coords(7,1)),squeeze(Coords(7,2)),squeeze(Coords(7,3)),'sb','LineWidth',1.5), hold on
scatter3(squeeze(-Coords(8,1)),squeeze(Coords(8,2)),squeeze(Coords(8,3)),'sc','LineWidth',1.5), hold on
xlabel('Dimension 1 (a.u.)')
ylabel('Dimension 2 (a.u.)')
zlabel('Dimension 3 (a.u.)')
xlim([-1.2 1.2])
ylim([-1.2 1.2])
zlim([-1.2 1.2])
pbaspect([1 1 1])
set(gca,'TickDir','out')
title('Fig. 5A, left')
%% figure 5A middle
figure
scatter(squeeze(-Coords(1,1)),squeeze(Coords(1,2)),'xm','LineWidth',1.5), hold on
scatter(squeeze(-Coords(2,1)),squeeze(Coords(2,2)),'xg','LineWidth',1.5), hold on
scatter(squeeze(-Coords(3,1)),squeeze(Coords(3,2)),'xb','LineWidth',1.5), hold on
scatter(squeeze(-Coords(4,1)),squeeze(Coords(4,2)),'xc','LineWidth',1.5), hold on
scatter(squeeze(-Coords(5,1)),squeeze(Coords(5,2)),'sm','LineWidth',1.5), hold on
scatter(squeeze(-Coords(6,1)),squeeze(Coords(6,2)),'sg','LineWidth',1.5), hold on
scatter(squeeze(-Coords(7,1)),squeeze(Coords(7,2)),'sb','LineWidth',1.5), hold on
scatter(squeeze(-Coords(8,1)),squeeze(Coords(8,2)),'sc','LineWidth',1.5), hold on
xlabel('Dimension 1 (a.u.)')
ylabel('Dimension 2 (a.u.)')
xlim([-1.2 1.2])
ylim([-1.2 1.2])
pbaspect([1 1 1])
set(gca,'TickDir','out')
title('Fig. 5A, middle')
%% figure 5A right
figure
scatter(squeeze(Coords(1,2)),squeeze(Coords(1,3)),'xm','LineWidth',1.5), hold on
scatter(squeeze(Coords(2,2)),squeeze(Coords(2,3)),'xg','LineWidth',1.5), hold on
scatter(squeeze(Coords(3,2)),squeeze(Coords(3,3)),'xb','LineWidth',1.5), hold on
scatter(squeeze(Coords(4,2)),squeeze(Coords(4,3)),'xc','LineWidth',1.5), hold on
scatter(squeeze(Coords(5,2)),squeeze(Coords(5,3)),'sm','LineWidth',1.5), hold on
scatter(squeeze(Coords(6,2)),squeeze(Coords(6,3)),'sg','LineWidth',1.5), hold on
scatter(squeeze(Coords(7,2)),squeeze(Coords(7,3)),'sb','LineWidth',1.5), hold on
scatter(squeeze(Coords(8,2)),squeeze(Coords(8,3)),'sc','LineWidth',1.5), hold on
xlabel('Dimension 2 (a.u.)')
ylabel('Dimension 3 (a.u.)')
xlim([-1.2 1.2])
ylim([-1.2 1.2])
pbaspect([1 1 1])
set(gca,'TickDir','out')
title('Fig. 5A, right')

