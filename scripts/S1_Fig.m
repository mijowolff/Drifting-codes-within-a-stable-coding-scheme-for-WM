clear all;
close all;
clc;
%% Supplemental figure 2 from Drifting codes (2019) Wolff et al.
cd('D:\Drifting codes\upload files') %path to main dir.
addpath(genpath(cd))

% channels of interest
test_chans = {'P7';'P5';'P3';'P1';'Pz';'P2';'P4';'P6';'P8';'PO7';'PO3';'POz';'PO4';'PO8';'O2';'O1';'Oz'};

% setting up the orientation spaces to group the item orientations into
bin_width=pi/8;
ang_steps=8; % 8 different orientation spaces to loop over
angspace_temp=(-pi:bin_width:pi)';
angspace_temp(end)=[];
for as=1:ang_steps
    angspaces(:,as)=angspace_temp+(as-1)*bin_width/ang_steps;
end
% number of repetitions (partitioning for cross validation and subsampling
% are always random
reps=100;

ds=5; % downsample continuous data (5 = 10 ms for 500 Hz)
gs=8; % gaussian smoothing width, SD in timepoints (8=16ms SD)
toi=[-0.001 1.6001]; % time window of interest, relative to impulse 1 onset
bl=[-.201 0]; % baseline
n_folds=8; % number of folds for cross-validation

do_decoding=0; %1=run the decodng (takes very long time!) or 0= load in previous output
%%
if do_decoding
    for sub=1:26 % loop over subjects
        fprintf(['Doing ' num2str(sub) '\n'])
        
        load(['dat_imp12_' num2str(sub) '.mat']);
        
        % remove trials based on bad performance (>circular SDs)
        report_err=circ_ang2rad(Results(:,6).*2);
        [~,SD]=circ_std(report_err);
        incl_ng=find(abs(circ_dist(report_err,circ_mean(report_err)))<(3*SD));
        
        % trials to include (based on performance and previously marked
        % "bad" trials
        incl_i=intersect(incl_ng,setdiff(1:size(Results,1), ft_imp12.bad_trials));
        
        bdat=mean(ft_imp12.trial(incl_i,ismember(ft_imp12.label,test_chans),ft_imp12.time>bl(1)&ft_imp12.time<=bl(2)),3);
        
        dat_temp=ft_imp12.trial(incl_i,ismember(ft_imp12.label,test_chans),ft_imp12.time>toi(1)&ft_imp12.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,bdat);
        dat_temp=smoothdata(dat_temp,3,'gaussian',gs*5);
        dat_imp12=dat_temp(:,:,1:ds:end);
        
        time=ft_imp12.time(ft_imp12.time>toi(1)&ft_imp12.time<=toi(2));
        time=time(1:ds:end);
        %%
        cue_cond=Results(:,3);
        cued_rad=Results(:,4).*2;
        uncued_rad=Results(:,5).*2;
        %%
        cued_left_cos_imp12=nan(ang_steps,reps,length(time),length(time));
        cued_right_cos_imp12=nan(ang_steps,reps,length(time),length(time));
        %%
        for a=1:ang_steps % loop through each orientation space
            
            % replace orientation values to the closes value of the current
            % orientation space
            
            ang_bin_temp=repmat(angspaces(:,a),[1 length(Results)]);
            
            [~,ind]= min(abs(circ_dist2(angspaces(:,a),cued_rad)));
            bin_cued=ang_bin_temp(ind)';
            bin_cued_i=bin_cued(incl_i,1);
            
            [~,ind]= min(abs(circ_dist2(angspaces(:,a),uncued_rad)));
            bin_uncued=ang_bin_temp(ind)';
            bin_uncued_i=bin_uncued(incl_i,1);
            
            for r=1:reps % repeat decoding "reps" times
                %%
                % cued left
                [distance_cos] = mahal_theta_kfold_basis_b_ct(dat_imp12(cue_cond(incl_i)==1,:,:),bin_cued_i(cue_cond(incl_i)==1),n_folds);
                cued_left_cos_imp12(a,r,:,:)=mean(distance_cos,1);
                
                % cued right
                [distance_cos] = mahal_theta_kfold_basis_b_ct(dat_imp12(cue_cond(incl_i)==2,:,:),bin_cued_i(cue_cond(incl_i)==2),n_folds);
                cued_right_cos_imp12(a,r,:,:)=mean(distance_cos,1);
            end
        end
        cos_imp12_cued(sub,:,:)=squeeze(mean(mean(cat(2,cued_left_cos_imp12,cued_right_cos_imp12),1),2));
    end
else
    load('S2_Fig_results.mat')
end
%% make cross-temporal symmetrical
% we have no hypothesis about asymmetries between testing/training between
% impulse 1/2 and vice versa...
for sub=1:26
    cos_imp12_cued_s(sub,:,:)=(squeeze(cos_imp12_cued(sub,:,:))+permute(squeeze(cos_imp12_cued(sub,:,:)),[2 1]))./2;
end
%%
figure
hold all
imagesc(time,time,squeeze(mean(cos_imp12_cued_s,1)),[0 0.0015])
pbaspect([1,1,1])
line([.8 .8],[min(time) max(time)],'Color','black')
line([min(time) max(time)],[.8 .8],'Color','black')
xlim([0 1.6])
ylim([0 1.6])
xlabel('Time (s) - rel. to impulse 1')
ylabel('Time (s) - rel. to impulse 1')
set(gca,'TickDir','out')
colormap 'hot'
colorbar
