clear all;
close all;
clc;
%% Supplemental figure 3A, from Drifting codes (2020) Wolff et al.
cd('D:\Drifting codes\upload files') % path to main dir.
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
    for sub=1:24
        fprintf(['Doing ' num2str(sub) '\n'])
        
        load(['dat_2015_' num2str(sub) '.mat']);
        
        % trials to include (based on performance and previously marked
        % "bad" trials
        incl_i=setdiff(1:size(Results,1), ft_imp.bad_trials);
        
        % extract good trials, channels and time window of interest of
        % impulse epoch and reformat
        dat_temp=ft_imp.trial(incl_i,ismember(ft_imp.label,test_chans),ft_imp.time>toi(1)&ft_imp.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3)); % take relative baseline
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard'); % downsample
        dat_temp=dat_temp(:,:,1:span:end);
        dat_imp=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]); %combine channel and time dimensions
        
        clear ft_imp
        %%
        onset_cond=Results(incl_i,2);
        item_rad=Results(incl_i,1).*2;
        
        % replace orientation values to the closes value of the current
        % orientation space
        ang_bin_temp=repmat(angspace,[1 length(item_rad)]);
        [~,ind]= min(abs(circ_dist2(angspace,item_rad)));
        bin_item=ang_bin_temp(ind)';
        
        bin_item_early=bin_item(onset_cond==1,1);
        bin_item_late=bin_item(onset_cond==2,1);
        
        dat_early=dat_imp(onset_cond==1,:,:);
        dat_late=dat_imp(onset_cond==2,:,:);
        clear dat_imp
        %%
        % compute covariance matrices for mahalanobis distances
        sigma=covdiag(cat(1,dat_late,dat_early));
        
         % get minimum number of trials for subsampling
        out = [angspace',histc(bin_item_early,angspace')];
        min_cond_early=min(out(:,2));
        out = [angspace',histc(bin_item_late,angspace')];
        min_cond_late=min(out(:,2));
        min_cond=min(cat(1,min_cond_early,min_cond_late));
        dists_temp=nan(reps,8,8);
        for r=1:reps
            dat_early_av=[]; dat_late_av=[];
            for ang=angspace
                trls=randsample(find(bin_item_late==ang),min_cond);
                dat_temp=mean(dat_late(trls,:),1);
                dat_late_av=cat(1,dat_late_av,dat_temp);
                trls=randsample(find(bin_item_early==ang),min_cond);
                dat_temp=mean(dat_early(trls,:),1);
                dat_early_av=cat(1,dat_early_av,dat_temp);
            end
            % get all pairwise distances between orientations and locations
            dists_temp(r,:,:)=squareform(pdist(cat(1,dat_late_av,dat_early_av),'mahalanobis',sigma));
        end
        dists(sub,:,:)=squeeze(mean(dists_temp,1));
    end
else
    load('S3A_Fig_results.mat')
end
Coords=cmdscale(squeeze(mean(dists,1)));
%% Suppl. figure 3A left
figure
scatter3(squeeze(Coords(1,3)),squeeze(Coords(1,2)),squeeze(Coords(1,1)),'dm','LineWidth',1.5), hold on
scatter3(squeeze(Coords(2,3)),squeeze(Coords(2,2)),squeeze(Coords(2,1)),'dg','LineWidth',1.5), hold on
scatter3(squeeze(Coords(3,3)),squeeze(Coords(3,2)),squeeze(Coords(3,1)),'db','LineWidth',1.5), hold on
scatter3(squeeze(Coords(4,3)),squeeze(Coords(4,2)),squeeze(Coords(4,1)),'dc','LineWidth',1.5), hold on
scatter3(squeeze(Coords(5,3)),squeeze(Coords(5,2)),squeeze(Coords(5,1)),'om','LineWidth',1.5), hold on
scatter3(squeeze(Coords(6,3)),squeeze(Coords(6,2)),squeeze(Coords(6,1)),'og','LineWidth',1.5), hold on
scatter3(squeeze(Coords(7,3)),squeeze(Coords(7,2)),squeeze(Coords(7,1)),'ob','LineWidth',1.5), hold on
scatter3(squeeze(Coords(8,3)),squeeze(Coords(8,2)),squeeze(Coords(8,1)),'oc','LineWidth',1.5), hold on
xlabel('Dimension 3 (a.u.)')
ylabel('Dimension 2 (a.u.)')
zlabel('Dimension 1 (a.u.)')
xlim([-1.2 1.2])
ylim([-1.2 1.2])
zlim([-1.2 1.2])
pbaspect([1 1 1])
set(gca,'TickDir','out')
title('Suppl. fig. 2A, left')
%% Suppl. figure 3A middle
figure
scatter(squeeze(Coords(1,3)),squeeze(Coords(1,1)),'dm','LineWidth',1.5), hold on
scatter(squeeze(Coords(2,3)),squeeze(Coords(2,1)),'dg','LineWidth',1.5), hold on
scatter(squeeze(Coords(3,3)),squeeze(Coords(3,1)),'db','LineWidth',1.5), hold on
scatter(squeeze(Coords(4,3)),squeeze(Coords(4,1)),'dc','LineWidth',1.5), hold on
scatter(squeeze(Coords(5,3)),squeeze(Coords(5,1)),'om','LineWidth',1.5), hold on
scatter(squeeze(Coords(6,3)),squeeze(Coords(6,1)),'og','LineWidth',1.5), hold on
scatter(squeeze(Coords(7,3)),squeeze(Coords(7,1)),'ob','LineWidth',1.5), hold on
scatter(squeeze(Coords(8,3)),squeeze(Coords(8,1)),'oc','LineWidth',1.5), hold on
xlabel('Dimension 3 (a.u.)')
ylabel('Dimension 1 (a.u.)')
xlim([-1.2 1.2])
ylim([-1.2 1.2])
pbaspect([1 1 1])
set(gca,'TickDir','out')
title('Suppl. fig. 2A, middle')
%% Suppl. figure 3A right
figure
scatter(squeeze(-Coords(1,2)),squeeze(Coords(1,1)),'dm','LineWidth',1.5), hold on
scatter(squeeze(-Coords(2,2)),squeeze(Coords(2,1)),'dg','LineWidth',1.5), hold on
scatter(squeeze(-Coords(3,2)),squeeze(Coords(3,1)),'db','LineWidth',1.5), hold on
scatter(squeeze(-Coords(4,2)),squeeze(Coords(4,1)),'dc','LineWidth',1.5), hold on
scatter(squeeze(-Coords(5,2)),squeeze(Coords(5,1)),'om','LineWidth',1.5), hold on
scatter(squeeze(-Coords(6,2)),squeeze(Coords(6,1)),'og','LineWidth',1.5), hold on
scatter(squeeze(-Coords(7,2)),squeeze(Coords(7,1)),'ob','LineWidth',1.5), hold on
scatter(squeeze(-Coords(8,2)),squeeze(Coords(8,1)),'oc','LineWidth',1.5), hold on
xlabel('Dimension 2 (a.u.)')
ylabel('Dimension 1 (a.u.)')
xlim([-1.2 1.2])
ylim([-1.2 1.2])
pbaspect([1 1 1])
set(gca,'TickDir','out')
title('Suppl. fig. 2A, right')

