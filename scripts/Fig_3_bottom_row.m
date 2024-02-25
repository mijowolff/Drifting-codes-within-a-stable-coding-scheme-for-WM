clear all;
close all;
clc;
%% Figure 3, bottom row, from Drifting codes (2020) Wolff et al.
cd('D:\Drifting codes\upload files') %path to main dir.
addpath(genpath(cd))

% path to fieldtrip (needs to be installed separately, necessary for
% plotting
addpath('fieldtrip path') 
ft_defaults

all_chans_left = {'Fpz','Fz','FCz','Cz','CPz','Pz','POz','Oz','Fp1','AF3','F1','FC1','C1',...
    'CP1','P1','PO3','O1','AF7','F7','FT7','T7','TP7','P7','PO7','F5','F3','FC5','FC3',...
    'C5','C3','CP5','CP3','P5','P3','Fp2','AF4','F2','FC2','C2','CP2','P2','PO4','O2',...
    'AF8','F8','FT8','T8','TP8','P8','PO8','F4','F6','FC4','FC6','C4','C6','CP4','CP6','P4','P6'};
all_chans_right = {'Fpz','Fz','FCz','Cz','CPz','Pz','POz','Oz','Fp2','AF4','F2','FC2','C2',...
    'CP2','P2','PO4','O2','AF8','F8','FT8','T8','TP8','P8','PO8','F6','F4','FC6','FC4',...
    'C6','C4','CP6','CP4','P6','P4','Fp1','AF3','F1','FC1','C1','CP1','P1','PO3','O1',...
    'AF7','F7','FT7','T7','TP7','P7','PO7','F3','F5','FC3','FC5','C3','C5','CP3','CP5','P3','P5'};

left_chans = {'P7';'P5';'P3';'P1';'PO7';'PO3';'O1'};
right_chans = {'P2';'P4';'P6';'P8';'PO4';'PO8';'O2'};

[tf,loc] = ismember(all_chans_left,all_chans_right);
bin_width=pi/8;
ang_steps=8;
angspace_temp=(-pi:bin_width:pi)';
angspace_temp(end)=[];
for as=1:ang_steps
    angspaces(:,as)=angspace_temp+(as-1)*bin_width/ang_steps;
end
reps=100; % number of repeats for random subsampling and cross-validation
span=5; %width of each segment
n_folds=8; % number of folds for cross-validation
toi=[.1 .4001];
perms=100000; % number of permutations for stats and confidence intervals
n_neigh=2; % number of channel neighbours to be included
do_decoding=0; %1=run the decodng (takes long time!) or 0= load in previous output
%%
if do_decoding
    for sub=1:26
        fprintf(['Doing ' num2str(sub) '\n'])
        
        load(['dat_' num2str(sub) '.mat']);
        
        report_err=circ_ang2rad(Results(:,6).*2);
        [~,SD]=circ_std(report_err);
        mu=circ_mean(report_err);
        incl_ng=find(abs(circ_dist(report_err,mu))<(3*SD));
        
        elec_dists=squareform(pdist(ft_mems.elec.chanpos(:,:)));
        [B,I_chans] = mink(elec_dists,n_neigh+1,1);
        
        incl_i=intersect(incl_ng,setdiff(1:size(Results,1), ft_imp1.bad_trials));
        incl_m=intersect(incl_ng,setdiff(1:size(Results,1), ft_mems.bad_trials));
        
        dat_temp=ft_mems.trial(incl_m,:,ft_mems.time>toi(1)&ft_mems.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3));
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard');
        dat_mem=dat_temp(:,:,1:span:end);
        
        clear ft_mems
        
        dat_temp=ft_imp1.trial(incl_i,:,ft_imp1.time>toi(1)&ft_imp1.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3));
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard');
        dat_imp1=dat_temp(:,:,1:span:end);
        
        clear ft_imp1
        
        dat_temp=ft_imp2.trial(incl_i,:,ft_imp2.time>toi(1)&ft_imp2.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3));
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard');
        dat_imp2=dat_temp(:,:,1:span:end);
        
        clear ft_imp2
        %%
        item_left=Results(:,1).*2;
        item_right=Results(:,2).*2;
        cue_cond=Results(:,3);
        cued_rad=Results(:,4).*2;
        uncued_rad=Results(:,5).*2;
        %%
        left_cos_mem=nan(ang_steps,length(I_chans),reps);
        right_cos_mem=nan(ang_steps,length(I_chans),reps);
        
        cued_left_cos_imp1=nan(ang_steps,length(I_chans),reps);
        cued_right_cos_imp1=nan(ang_steps,length(I_chans),reps);
        
        uncued_left_cos_imp1=nan(ang_steps,length(I_chans),reps);
        uncued_right_cos_imp1=nan(ang_steps,length(I_chans),reps);
        
        cued_left_cos_imp2=cued_left_cos_imp1;
        cued_right_cos_imp2=cued_right_cos_imp1;
        
        uncued_left_cos_imp2=uncued_left_cos_imp1;
        uncued_right_cos_imp2=uncued_right_cos_imp1;
        %%
        for a=1:ang_steps
            
            ang_bin_temp=repmat(angspaces(:,a),[1 length(Results)]);
            
            [~,ind]= min(abs(circ_dist2(angspaces(:,a),item_left)));
            bin_left=ang_bin_temp(ind)';
            bin_left_m=bin_left(incl_m,1);
            
            [~,ind]= min(abs(circ_dist2(angspaces(:,a),item_right)));
            bin_right=ang_bin_temp(ind)';
            bin_right_m=bin_right(incl_m,1);
            
            [~,ind]= min(abs(circ_dist2(angspaces(:,a),cued_rad)));
            bin_cued=ang_bin_temp(ind)';
            bin_cued_i=bin_cued(incl_i,1);
            
            [~,ind]= min(abs(circ_dist2(angspaces(:,a),uncued_rad)));
            bin_uncued=ang_bin_temp(ind)';
            bin_uncued_i=bin_uncued(incl_i,1);
            %%
            for ch=1:length(I_chans)
                dat_temp=dat_mem(:,I_chans(:,ch),:);
                dat_temp_mem=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]);
                
                dat_temp=dat_imp1(:,I_chans(:,ch),:);
                dat_temp_imp1=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]);
                
                dat_temp=dat_imp2(:,I_chans(:,ch),:);
                dat_temp_imp2=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]);
                %%
                for r=1:reps
                    %% item presentation
                    % left item
                    distance_cos = mahal_theta_kfold_basis_b(dat_temp_mem,bin_left_m,n_folds);
                    left_cos_mem(a,ch,r)=mean(distance_cos,2);
                    
                    % right item
                    distance_cos = mahal_theta_kfold_basis_b(dat_temp_mem,bin_right_m,n_folds);
                    right_cos_mem(a,ch,r)=mean(distance_cos,2);
                    %% impulse 1
                    % cued left
                    distance_cos = mahal_theta_kfold_basis_b(dat_temp_imp1(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1),n_folds);
                    cued_left_cos_imp1(a,ch,r)=mean(distance_cos,2);
                    
                    % cued right
                    distance_cos = mahal_theta_kfold_basis_b(dat_temp_imp1(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2),n_folds);
                    cued_right_cos_imp1(a,ch,r)=mean(distance_cos,2);
                    
                    % uncued left
                    distance_cos = mahal_theta_kfold_basis_b(dat_temp_imp1(cue_cond(incl_i)==2,:),bin_uncued_i(cue_cond(incl_i)==2),n_folds);
                    uncued_left_cos_imp1(a,ch,r)=mean(distance_cos,2);
                    
                    % uncued right
                    distance_cos = mahal_theta_kfold_basis_b(dat_temp_imp1(cue_cond(incl_i)==1,:),bin_uncued_i(cue_cond(incl_i)==1),n_folds);
                    uncued_right_cos_imp1(a,ch,r)=mean(distance_cos,2);
                    %% impulse 2
                    % cued left
                    distance_cos = mahal_theta_kfold_basis_b(dat_temp_imp2(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1),n_folds);
                    cued_left_cos_imp2(a,ch,r)=mean(distance_cos,2);
                    
                    % cued right
                    distance_cos = mahal_theta_kfold_basis_b(dat_temp_imp2(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2),n_folds);
                    cued_right_cos_imp2(a,ch,r)=mean(distance_cos,2);
                    
                    % uncued left
                    distance_cos = mahal_theta_kfold_basis_b(dat_temp_imp2(cue_cond(incl_i)==2,:),bin_uncued_i(cue_cond(incl_i)==2),n_folds);
                    uncued_left_cos_imp2(a,ch,r)=mean(distance_cos,2);
                    
                    % uncued right
                    distance_cos = mahal_theta_kfold_basis_b(dat_temp_imp2(cue_cond(incl_i)==1,:),bin_uncued_i(cue_cond(incl_i)==1),n_folds);
                    uncued_right_cos_imp2(a,ch,r)=mean(distance_cos,2);
                end
            end
        end
        
        left_temp=squeeze(mean(mean(left_cos_mem,1),3));
        right_temp=squeeze(mean(mean(right_cos_mem,1),3));
        cos_mem_both(sub,:)=(left_temp+right_temp(loc))./2;
        
        left_temp=squeeze(mean(mean(cued_left_cos_imp1,1),3));
        right_temp=squeeze(mean(mean(cued_right_cos_imp1,1),3));
        cos_imp1_cued(sub,:)=(left_temp+right_temp(loc))./2;
        
        left_temp=squeeze(mean(mean(uncued_left_cos_imp1,1),3));
        right_temp=squeeze(mean(mean(uncued_right_cos_imp1,1),3));
        cos_imp1_uncued(sub,:)=(left_temp+right_temp(loc))./2;
        
        left_temp=squeeze(mean(mean(cued_left_cos_imp2,1),3));
        right_temp=squeeze(mean(mean(cued_right_cos_imp2,1),3));
        cos_imp2_cued(sub,:)=(left_temp+right_temp(loc))./2;
        
        left_temp=squeeze(mean(mean(uncued_left_cos_imp2,1),3));
        right_temp=squeeze(mean(mean(uncued_right_cos_imp2,1),3));
        cos_imp2_uncued(sub,:)=(left_temp+right_temp(loc))./2;
    end
    %% insert decoding values into fieldtrip data file for plotting
    cfg = [];
    cfg.latency= [0 0.002];
    cfg.channel=all_chans_left;
    ft= ft_selectdata(cfg, ft_imp1);
    cfg = [];
    chanst = ft_timelockanalysis(cfg, ft);
    chanst.label=all_chans_left;
    
    mem_both_chans=chanst;
    imp1_cued_chans=chanst;
    imp1_uncued_chans=chanst;
    imp2_cued_chans=chanst;
    imp2_uncued_chans=chanst;
    %
    mem_both_chans.avg=repmat(mean(cos_mem_both,1)',[1 2]);
    imp1_cued_chans.avg=repmat(mean(cos_imp1_cued,1)',[1 2]);
    imp1_uncued_chans.avg=repmat(mean(cos_imp1_uncued,1)',[1 2]);
    imp2_cued_chans.avg=repmat(mean(cos_imp2_cued,1)',[1 2]);
    imp2_uncued_chans.avg=repmat(mean(cos_imp2_uncued,1)',[1 2]);
    %% extract ipsi and contra posterior channels for significance testing
    chans_right_inds=ismember(ft_imp1.label,right_chans);
    chans_left_inds=ismember(ft_imp1.label,left_chans);
    
    mem_chans_contr=mean(cos_mem_both(:,chans_right_inds),2);
    mem_chans_ips=mean(cos_mem_both(:,chans_left_inds),2);
    
    imp1_cued_chans_contr=mean(cos_imp1_cued(:,chans_right_inds),2);
    imp1_cued_chans_ips=mean(cos_imp1_cued(:,chans_left_inds),2);
    
    imp2_cued_chans_contr=mean(cos_imp2_cued(:,chans_right_inds),2);
    imp2_cued_chans_ips=mean(cos_imp2_cued(:,chans_left_inds),2);
    
    imp1_uncued_chans_contr=mean(cos_imp1_uncued(:,chans_right_inds),2);
    imp1_uncued_chans_ips=mean(cos_imp1_uncued(:,chans_left_inds),2);
    
    imp2_uncued_chans_contr=mean(cos_imp2_uncued(:,chans_right_inds),2);
    imp2_uncued_chans_ips=mean(cos_imp2_uncued(:,chans_left_inds),2);
else
    load('Fig_3_bottom_results.mat')
end
%%
fhandle=figure;
set(fhandle, 'Position', [1, 1, 1800, 400]);
subplot(1,5,1)
cfg = [];
cfg.zlim = [-.0 .01];
cfg.layout = 'elec1005.lay';
cfg.colorbar = 'yes';
cfg.highlight = 'off';
cfg.marker='off';
ft_topoplotER(cfg,mem_both_chans);
colormap('hot')
title('Memory items')

subplot(1,5,2)
cfg = [];
cfg.zlim = [-.0 .001];
cfg.layout = 'elec1005.lay';
cfg.colorbar = 'yes';
cfg.marker='off';
ft_topoplotER(cfg,imp1_cued_chans);
colormap('hot')
title('Impulse 1 - cued')

subplot(1,5,3)
cfg = [];
cfg.zlim = [-.0 .001]; 
cfg.layout = 'elec1005.lay';
cfg.colorbar = 'yes';
cfg.marker='off';
ft_topoplotER(cfg,imp1_uncued_chans);
colormap('hot')
title('Impulse 1 - uncued')

subplot(1,5,4)
cfg = [];
cfg.zlim = [-.0 .001];
cfg.layout = 'elec1005.lay';
cfg.colorbar = 'yes';
cfg.marker='off';
ft_topoplotER(cfg,imp2_cued_chans);
colormap('hot')
title('Impulse 2 - cued')

subplot(1,5,5)
cfg = [];
cfg.zlim = [-.0 .001];
cfg.layout = 'elec1005.lay';
cfg.colorbar = 'yes';
cfg.marker='off';
ft_topoplotER(cfg,imp2_uncued_chans);
colormap('hot')
title('Impulse 2 - uncued')
%% statistics on lateralization 
use_NT=1; % 1=use prepared null distributions from null deocoder, 0=make null distributions by shuffling data
if use_NT
    % make t-values
    mem_chans_diff_t=FastTtest(mem_chans_contr-mem_chans_ips);
    imp1_cued_chans_diff_t=FastTtest(imp1_cued_chans_contr-imp1_cued_chans_ips);
    imp2_cued_chans_diff_t=FastTtest(imp2_cued_chans_contr-imp2_cued_chans_ips);
    imp1_uncued_chans_diff_t=FastTtest(imp1_uncued_chans_contr-imp1_uncued_chans_ips);
    imp2_uncued_chans_diff_t=FastTtest(imp2_uncued_chans_contr-imp2_uncued_chans_ips);
    
    load('Fig_3_bottom_NULL_T.mat')
    
    p_mem_diff=FastPvalue(mem_chans_diff_t,mem_chans_diff_NT,2);
    p_imp1_cued_diff=FastPvalue(imp1_cued_chans_diff_t,imp1_cued_chans_diff_NT,2);
    p_imp2_cued_diff=FastPvalue(imp2_cued_chans_diff_t,imp2_cued_chans_diff_NT,2);
    p_imp1_uncued_diff=FastPvalue(imp1_uncued_chans_diff_t,imp1_uncued_chans_diff_NT,2);
    p_imp2_uncued_diff=FastPvalue(imp2_uncued_chans_diff_t,imp2_uncued_chans_diff_NT,2);
else
    p_mem_diff=GroupPermTest(mem_chans_contr-mem_chans_ips,perms,2,'t');
    p_imp1_cued_diff=GroupPermTest(imp1_cued_chans_contr-imp1_cued_chans_ips,perms,2,'t');
    p_imp2_cued_diff=GroupPermTest(imp2_cued_chans_contr-imp2_cued_chans_ips,perms,2,'t');
    p_imp1_uncued_diff=GroupPermTest(imp1_uncued_chans_contr-imp1_uncued_chans_ips,perms,2,'t');
    p_imp2_uncued_diff=GroupPermTest(imp2_uncued_chans_contr-imp2_uncued_chans_ips,perms,2,'t');
end    
