clear all;
close all;
clc;
%% Figure 7BC, from Drifting codes (2020) Wolff et al.
cd('D:\Drifting codes\upload files') %path to main dir.
addpath(genpath(cd))

% channels of interest
test_chans = {'P7';'P5';'P3';'P1';'Pz';'P2';'P4';'P6';'P8';'PO7';'PO3';'POz';'PO4';'PO8';'O2';'O1';'Oz'};

% setting up the orientation spaces to group the item orientations into
bin_width=pi/8;
ang_steps=8;
angspace=(-pi:bin_width:pi)';
angspace(end)=[];
for as=1:ang_steps
    angspaces(:,as)=angspace+(as-1)*bin_width/ang_steps;
end
reps=100; % number of repeats for random subsampling
span=5; % number of time-points to average over (5 = 10 ms for 500 Hz)
toi=[.1 .4001];
perms=100000; % number of permutations for stats and confidence intervals
do_decoding=0; %1=run the decodng (takes long time!) or 0= load in previous output
%%
if do_decoding
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
        incl_m=intersect(incl_ng,setdiff(1:size(Results,1), ft_mems.bad_trials));
        
        % extract good trials, channels and time window of interest of
        % memory epoch and reformat
        dat_temp=ft_mems.trial(incl_m,ismember(ft_mems.label,test_chans),ft_mems.time>toi(1)&ft_mems.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3)); % take relative baseline
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard'); % downsample 
        dat_temp=dat_temp(:,:,1:span:end);
        dat_mem=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]); %combine channel and time dimensions
        
        clear ft_mems
        
        % extract good trials, channels and time window of interest of
        % impulse 1 epoch and reformat
        dat_temp=ft_imp1.trial(incl_i,ismember(ft_imp1.label,test_chans),ft_imp1.time>toi(1)&ft_imp1.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3));
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard');
        dat_temp=dat_temp(:,:,1:span:end);
        dat_imp1=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]);
        
        clear ft_imp1
        
        % extract good trials, channels and time window of interest of
        % impulse 2 epoch and reformat
        dat_temp=ft_imp2.trial(incl_i,ismember(ft_imp2.label,test_chans),ft_imp2.time>toi(1)&ft_imp2.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3));
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard');
        dat_temp=dat_temp(:,:,1:span:end);
        dat_imp2=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]);
        
        clear ft_imp2
        
        dat_imps=(dat_imp1+dat_imp2)./2; % combine impulses for training
        %%
        cue_cond=Results(:,3);
        cued_rad=Results(:,4).*2;
        
        report_err_m=report_err(incl_m,1);
        report_err_i=report_err(incl_i,1);
        %% preallocate decoding output matrices for each item and epoch
        cued_left_dists_mem_ccw=nan(16,ang_steps,reps);cued_right_dists_mem_ccw=nan(16,ang_steps,reps);
        cued_left_dists_mem_cw=nan(16,ang_steps,reps);cued_right_dists_mem_cw=nan(16,ang_steps,reps);
        cued_left_dists_imp1_ccw=nan(16,ang_steps,reps);cued_right_dists_imp1_ccw=nan(16,ang_steps,reps);
        cued_left_dists_imp1_cw=nan(16,ang_steps,reps);cued_right_dists_imp1_cw=nan(16,ang_steps,reps);
        cued_left_dists_imp2_ccw=nan(16,ang_steps,reps);cued_right_dists_imp2_ccw=nan(16,ang_steps,reps);
        cued_left_dists_imp2_cw=nan(16,ang_steps,reps);cued_right_dists_imp2_cw=nan(16,ang_steps,reps);
        %%
        for a=1:ang_steps
            
            % replace orientation values to the closes value of the current
            % orientation space
            ang_bin_temp=repmat(angspaces(:,a),[1 length(Results)]);
            [~,ind]= min(abs(circ_dist2(angspaces(:,a),cued_rad)));
            bin_cued=ang_bin_temp(ind)';
            bin_cued_m=bin_cued(incl_m,1);
            bin_cued_i=bin_cued(incl_i,1);
            
            %% median split data into ccw and cw parts, making sure that the same
            % number of trials of each orientation is in each split
            [dat_mem_ccw_left,dat_mem_cw_left,bin_m_ccw_left,bin_m_cw_left]=report_split_dat_func(dat_mem(cue_cond(incl_m)==1,:),report_err_m(cue_cond(incl_m)==1,1),bin_cued_m(cue_cond(incl_m)==1));
            [dat_mem_ccw_right,dat_mem_cw_right,bin_m_ccw_right,bin_m_cw_right]=report_split_dat_func(dat_mem(cue_cond(incl_m)==2,:),report_err_m(cue_cond(incl_m)==2,1),bin_cued_m(cue_cond(incl_m)==2));
            
            [dat_imp1_ccw_left,dat_imp1_cw_left,bin_i_ccw_left,bin_i_cw_left]=report_split_dat_func(dat_imp1(cue_cond(incl_i)==1,:),report_err_i(cue_cond(incl_i)==1,1),bin_cued_i(cue_cond(incl_i)==1));
            [dat_imp1_ccw_right,dat_imp1_cw_right,bin_i_ccw_right,bin_i_cw_right]=report_split_dat_func(dat_imp1(cue_cond(incl_i)==2,:),report_err_i(cue_cond(incl_i)==2,1),bin_cued_i(cue_cond(incl_i)==2));
            
            [dat_imp2_ccw_left,dat_imp2_cw_left]=report_split_dat_func(dat_imp2(cue_cond(incl_i)==1,:),report_err_i(cue_cond(incl_i)==1,1),bin_cued_i(cue_cond(incl_i)==1));
            [dat_imp2_ccw_right,dat_imp2_cw_right]=report_split_dat_func(dat_imp2(cue_cond(incl_i)==2,:),report_err_i(cue_cond(incl_i)==2,1),bin_cued_i(cue_cond(incl_i)==2));
            
            % impulse epochs are trained on both, but tested separately
            [dat_imps_ccw_left,dat_imps_cw_left]=report_split_dat_func(dat_imps(cue_cond(incl_i)==1,:),report_err_i(cue_cond(incl_i)==1,1),bin_cued_i(cue_cond(incl_i)==1));
            [dat_imps_ccw_right,dat_imps_cw_right]=report_split_dat_func(dat_imps(cue_cond(incl_i)==2,:),report_err_i(cue_cond(incl_i)==2,1),bin_cued_i(cue_cond(incl_i)==2));
            %%
            for r=1:reps % repeat decoding "reps" times
                %% item presentation
                % left, ccw
                [~,distances] = mahal_theta_basis_b_ind(dat_mem_ccw_left,bin_m_ccw_left,dat_mem_cw_left,bin_m_cw_left);
                cued_left_dists_mem_ccw(:,a,r)=mean(distances,2);
                
                % left, cw
                [~,distances] = mahal_theta_basis_b_ind(dat_mem_cw_left,bin_m_cw_left,dat_mem_ccw_left,bin_m_ccw_left);
                cued_left_dists_mem_cw(:,a,r)=mean(distances,2);
                
                % right, ccw
                [~,distances] = mahal_theta_basis_b_ind(dat_mem_ccw_right,bin_m_ccw_right,dat_mem_cw_right,bin_m_cw_right);
                cued_right_dists_mem_ccw(:,a,r)=mean(distances,2);
                
                % right, cw
                [~,distances] = mahal_theta_basis_b_ind(dat_mem_cw_right,bin_m_cw_right,dat_mem_ccw_right,bin_m_ccw_right);
                cued_right_dists_mem_cw(:,a,r)=mean(distances,2);
                %% impulse 1
                % left, ccw
                [~,distances] = mahal_theta_basis_b_ind(dat_imp1_ccw_left,bin_i_ccw_left,dat_imps_cw_left,bin_i_cw_left);
                cued_left_dists_imp1_ccw(:,a,r)=mean(distances,2);
                
                % left, cw
                [~,distances] = mahal_theta_basis_b_ind(dat_imp1_cw_left,bin_i_cw_left,dat_imps_ccw_left,bin_i_ccw_left);
                cued_left_dists_imp1_cw(:,a,r)=mean(distances,2);
                
                % right, ccw
                [~,distances] = mahal_theta_basis_b_ind(dat_imp1_ccw_right,bin_i_ccw_right,dat_imps_cw_right,bin_i_cw_right);
                cued_right_dists_imp1_ccw(:,a,r)=mean(distances,2);
                
                % right, cw
                [~,distances] = mahal_theta_basis_b_ind(dat_imp1_cw_right,bin_i_cw_right,dat_imps_ccw_right,bin_i_ccw_right);
                cued_right_dists_imp1_cw(:,a,r)=mean(distances,2);
                %% impulse 2
                % left, ccw
                [~,distances] = mahal_theta_basis_b_ind(dat_imp2_ccw_left,bin_i_ccw_left,dat_imps_cw_left,bin_i_cw_left);
                cued_left_dists_imp2_ccw(:,a,r)=mean(distances,2);
                
                % left, cw
                [~,distances] = mahal_theta_basis_b_ind(dat_imp2_cw_left,bin_i_cw_left,dat_imps_ccw_left,bin_i_ccw_left);
                cued_left_dists_imp2_cw(:,a,r)=mean(distances,2);
                
                % right, ccw
                [~,distances] = mahal_theta_basis_b_ind(dat_imp2_ccw_right,bin_i_ccw_right,dat_imps_cw_right,bin_i_cw_right);
                cued_right_dists_imp2_ccw(:,a,r)=mean(distances,2);
                
                % right, cw
                [~,distances] = mahal_theta_basis_b_ind(dat_imp2_cw_right,bin_i_cw_right,dat_imps_ccw_right,bin_i_ccw_right);
                cued_right_dists_imp2_cw(:,a,r)=mean(distances,2);
            end
        end
        %% average over all reps and orientation spaces
        mem_ccw(sub,:)=(squeeze(mean(mean(cued_left_dists_mem_ccw,2),3))+squeeze(mean(mean(cued_right_dists_mem_ccw,2),3)))./2;
        mem_cw(sub,:)=(squeeze(mean(mean(cued_left_dists_mem_cw,2),3))+squeeze(mean(mean(cued_right_dists_mem_cw,2),3)))./2;
        
        imp1_ccw(sub,:)=(squeeze(mean(mean(cued_left_dists_imp1_ccw,2),3))+squeeze(mean(mean(cued_right_dists_imp1_ccw,2),3)))./2;
        imp1_cw(sub,:)=(squeeze(mean(mean(cued_left_dists_imp1_cw,2),3))+squeeze(mean(mean(cued_right_dists_imp1_cw,2),3)))./2;
        
        imp2_ccw(sub,:)=(squeeze(mean(mean(cued_left_dists_imp2_ccw,2),3))+squeeze(mean(mean(cued_right_dists_imp2_ccw,2),3)))./2;
        imp2_cw(sub,:)=(squeeze(mean(mean(cued_left_dists_imp2_cw,2),3))+squeeze(mean(mean(cued_right_dists_imp2_cw,2),3)))./2;
    end
else
    load('Fig_7BC_results')
end
%% summarize ccw and cw into single tuning curve 
temp_ccw=mem_ccw;temp_ccw(:,2:end)=flip(temp_ccw(:,2:end),2);
mem_bias=(mem_cw+temp_ccw)./2;

temp_ccw=imp1_ccw;temp_ccw(:,2:end)=flip(temp_ccw(:,2:end),2);
imp1_bias=(imp1_cw+temp_ccw)./2;

temp_ccw=imp2_ccw;temp_ccw(:,2:end)=flip(temp_ccw(:,2:end),2);
imp2_bias=(imp2_cw+temp_ccw)./2;
%% statistics on group level circular direction
mu=circ_mean(angspace,mean(mem_bias,1)');
mem_mu_av=circ_rad2ang(mu/2);
mu_sims=nan(1,perms);
for s=1:perms
    subs=randsample([1 2],26,'true');
    bias_temp=cat(1,mem_bias(subs==1,:),cat(2,mem_bias(subs==2,1),...
        fliplr(mem_bias(subs==2,2:end))));
    mu_sims(s)=circ_rad2ang(circ_mean(angspace,mean(bias_temp,1)')./2);
end
p_mem=sum(mu_sims>mem_mu_av)/perms;
%
mu=circ_mean(angspace,mean(imp1_bias,1)');
imp1_mu_av=circ_rad2ang(mu/2);
mu_sims=nan(1,perms);
for s=1:perms
    subs=randsample([1 2],26,'true');
    bias_temp=cat(1,imp1_bias(subs==1,:),cat(2,imp1_bias(subs==2,1),...
        fliplr(imp1_bias(subs==2,2:end))));
    mu_sims(s)=circ_rad2ang(circ_mean(angspace,mean(bias_temp,1)')./2);
end
p_imp1=sum(mu_sims>imp1_mu_av)/perms;
%
mu=circ_mean(angspace,mean(imp2_bias,1)');
imp2_mu_av=circ_rad2ang(mu/2);
mu_sims=nan(1,perms);
for s=1:perms
    subs=randsample([1 2],26,'true');
    bias_temp=cat(1,imp2_bias(subs==1,:),cat(2,imp2_bias(subs==2,1),...
        fliplr(imp2_bias(subs==2,2:end))));
    mu_sims(s)=circ_rad2ang(circ_mean(angspace,mean(bias_temp,1)')./2);
end
p_imp2=sum(mu_sims>imp2_mu_av)/perms;
%% statistics on subject level "asymmetry score"
mem_as=mean(mem_bias(:,10:16),2)-mean(mem_bias(:,2:8),2);
imp1_as=mean(imp1_bias(:,10:16),2)-mean(imp1_bias(:,2:8),2);
imp2_as=mean(imp2_bias(:,10:16),2)-mean(imp2_bias(:,2:8),2);
use_NT=1; % 1=use prepared null distributions from null deocoder, 0=make null distributions by shuffling data
if use_NT
    mem_as_t=FastTtest(mem_as);
    imp1_as_t=FastTtest(imp1_as);
    imp2_as_t=FastTtest(imp2_as);
    
    load('Fig_7C_NULL_T.mat')
    
    p_mem_as=FastPvalue(mem_as_t,mem_as_NT,1);
    p_imp1_as=FastPvalue(imp1_as_t,imp1_as_NT,1);
    p_imp2_as=FastPvalue(imp2_as_t,imp2_as_NT,1);
else
    p_mem_as=GroupPermTest(mem_as,perms,1,'t');
    p_imp1_as=GroupPermTest(imp1_as,perms,1,'t');
    p_imp2_as=GroupPermTest(imp2_as,perms,1,'t');
end
%% make tunning symmetrical for plotting (-90 and 90 degrees on either side, -90 = 90)
mem_cw2=cat(2,mem_cw,mem_cw(:,1));mem_ccw2=cat(2,mem_ccw,mem_ccw(:,1));
imp1_cw2=cat(2,imp1_cw,imp1_cw(:,1));imp1_ccw2=cat(2,imp1_ccw,imp1_ccw(:,1));
imp2_cw2=cat(2,imp2_cw,imp2_cw(:,1));imp2_ccw2=cat(2,imp2_ccw,imp2_ccw(:,1));

mem_bias2=cat(2,mem_bias,mem_bias(:,1));
imp1_bias2=cat(2,imp1_bias,imp1_bias(:,1));
imp2_bias2=cat(2,imp2_bias,imp2_bias(:,1));

angspace_ang=circ_rad2ang(angspace/2);
angspace_ang2=cat(1,angspace_ang,-angspace_ang(1));
%% make CIs using bootstrapping for plotting
ci_mem=bootci(perms,@mean,mem_bias2);
ci_imp1=bootci(perms,@mean,imp1_bias2);
ci_imp2=bootci(perms,@mean,imp2_bias2);
%% Figure 7B
fhandle=figure;
set(fhandle, 'Position', [100, 100, 1050, 350]);
subplot(1,3,1)
title('Memory array')
hold all
plot(angspace_ang2,mean(mem_bias2,1),'-ro','Color',[.5 0 .5],'LineWidth',2,'MarkerFaceColor',[.5 0 .5])
plot(angspace_ang2,ci_mem,'Color',[.5 0 .5],'LineWidth',.5, 'LineStyle', ':')
line(linspace(-0,0),linspace(-0.015,0.03001),'Color',[.5 0 .5])
line(linspace(mem_mu_av,mem_mu_av),linspace(-0.015,0.03001),'Color',[.5 0 .5])
xticks([angspace_ang2(1:2:end)])
set(gca,'TickDir','out')
ylim([-0.015 0.03001]);
xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')

subplot(1,3,2)
title('Impulse 1')
hold all
plot(angspace_ang2,mean(imp1_bias2,1),'-ro','Color',[.5 0 .5],'LineWidth',2,'MarkerFaceColor',[.5 0 .5])
plot(angspace_ang2,ci_imp1,'Color',[.5 0 .5],'LineWidth',.5, 'LineStyle', ':')
line(linspace(-0,0),linspace(-0.015,0.03001),'Color',[.5 0 .5])
line(linspace(imp1_mu_av,imp1_mu_av),linspace(-0.015,0.03001),'Color',[.5 0 .5])
xticks([angspace_ang2(1:2:end)])
set(gca,'TickDir','out')
ylim([-0.005 0.006]);xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')

subplot(1,3,3)
title('Impulse 2')
hold all
plot(angspace_ang2,mean(imp2_bias2(:,:),1),'-ro','Color',[.5 0 .5],'LineWidth',2,'MarkerFaceColor',[.5 0 .5])
plot(angspace_ang2,ci_imp2,'Color',[.5 0 .5],'LineWidth',.5, 'LineStyle', ':')
line(linspace(-0,0),linspace(-0.015,0.03001),'Color',[.5 0 .5])
line(linspace(imp2_mu_av,imp2_mu_av),linspace(-0.015,0.03001),'Color',[.5 0 .5])
xticks([angspace_ang2(1:2:end)])
set(gca,'TickDir','out')
ylim([-0.005 0.006]);
xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')
%% make C.I. error bars for plots
mu_sims=nan(1,perms);
for s=1:perms
    subs=randsample([1:26],26,'true');
    mu_sims(s)=circ_rad2ang(circ_mean(angspace,mean(mem_bias(subs,:),1)')./2);
end
sort_temp=sort(mu_sims);
sort_temp2=sort_temp(1:perms*0.025);
CI_mem(1)=sort_temp2(end);
sort_temp=sort(mu_sims,'descend');
sort_temp2=sort_temp(1:perms*0.025);
CI_mem(2)=sort_temp2(end);

mu_sims=nan(1,perms);
for s=1:perms
    subs=randsample([1:26],26,'true');
    mu_sims(s)=circ_rad2ang(circ_mean(angspace,mean(imp1_bias(subs,:),1)')./2);
end
sort_temp=sort(mu_sims);
sort_temp2=sort_temp(1:perms*0.025);
CI_imp1(1)=sort_temp2(end);
sort_temp=sort(mu_sims,'descend');
sort_temp2=sort_temp(1:perms*0.025);
CI_imp1(2)=sort_temp2(end);

mu_sims=nan(1,perms);
for s=1:perms
    subs=randsample([1:26],26,'true');
    mu_sims(s)=circ_rad2ang(circ_mean(angspace,mean(imp2_bias(subs,:),1)')./2);
end
sort_temp=sort(mu_sims);
sort_temp2=sort_temp(1:perms*0.025);
CI_imp2(1)=sort_temp2(end);
sort_temp=sort(mu_sims,'descend');
sort_temp2=sort_temp(1:perms*0.025);
CI_imp2(2)=sort_temp2(end);
%% Figure 7C (circular mean)
figure
pos=[.95 1 1.05];
hold all
plot(pos(1),mem_mu_av,'o','MarkerFaceColor',[.5 0 .5],'MarkerEdgeColor','none','MarkerSize',10)
plot(pos(2),imp1_mu_av,'o','MarkerFaceColor',[.5 0 .5],'MarkerEdgeColor','none','MarkerSize',10)
plot(pos(3),imp2_mu_av,'o','MarkerFaceColor',[.5 0 .5],'MarkerEdgeColor','none','MarkerSize',10)
plot([pos(1) pos(1)],CI_mem,'Color',[.5 0 .5],'LineWidth',3)
plot([pos(2) pos(2)],CI_imp1,'Color',[.5 0 .5],'LineWidth',3)
plot([pos(3) pos(3)],CI_imp2,'Color',[.5 0 .5],'LineWidth',3)
plot([.93 1.07],[0 0 ],'Color','k','LineWidth',.5,'LineStyle','-')
pbaspect([1 1 1])
xlim([.93 1.07]); ylim([-15 35])
set(gca,'XTick',[])
ylabel('Shift towards response')
set(gca,'TickDir','out')
%%
ci_mem_cw=bootci(perms,@mean,mem_cw2);
ci_mem_ccw=bootci(perms,@mean,mem_ccw2);
ci_imp1_cw=bootci(perms,@mean,imp1_cw2);
ci_imp1_ccw=bootci(perms,@mean,imp1_ccw2);
ci_imp2_cw=bootci(perms,@mean,imp2_cw2);
ci_imp2_ccw=bootci(perms,@mean,imp2_ccw2);
%%
CI_mem_as=bootci(perms,@mean,mem_as);
CI_imp1_as=bootci(perms,@mean,imp1_as);
CI_imp2_as=bootci(perms,@mean,imp2_as);
%% Figure 7C (asymmetry score)
figure
pos=[.95 1 1.05];
hold all
b1=boxplot([mem_as,imp1_as,imp2_as],...
    'positions',pos,'Widths',0.02,'Symbol','b.');
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
set(b1(:,1),'color',[.5 0 .5]);set(b1(:,2),'color',[.5 0 .5]);set(b1(:,3),'color',[.5 0 .5]);
plot(pos(1),mean(mem_as,1),'o','MarkerFaceColor',[.5 0 .5],'MarkerEdgeColor','none','MarkerSize',10)
plot(pos(2),mean(imp1_as,1),'o','MarkerFaceColor',[.5 0 .5],'MarkerEdgeColor','none','MarkerSize',10)
plot(pos(3),mean(imp2_as,1),'o','MarkerFaceColor',[.5 0 .5],'MarkerEdgeColor','none','MarkerSize',10)
plot([pos(1) pos(1)],CI_mem_as,'Color',[.5 0 .5],'LineWidth',3)
plot([pos(2) pos(2)],CI_imp1_as,'Color',[.5 0 .5],'LineWidth',3)
plot([pos(3) pos(3)],CI_imp2_as,'Color',[.5 0 .5],'LineWidth',3)
plot([.93 1.07],[0 0 ],'Color','k','LineWidth',.5,'LineStyle','-')
pbaspect([1 1 1])
xlim([.93 1.07]); 
ylim([-.006 .012])
set(gca,'XTick',[])
ylabel('Asymmetry score')
set(gca,'TickDir','out')
%% Figure 7B, insets
fhandle=figure;
set(fhandle, 'Position', [100, 100, 1050, 350]);
subplot(1,3,1)
title('Memory array')
hold all
plot(angspace_ang2,mean(mem_cw2,1),'-ro','Color',[0 .5 0],'LineWidth',2,'MarkerFaceColor',[0 .5 0])
plot(angspace_ang2,mean(mem_ccw2,1),'-ro','Color',[0 0 1],'LineWidth',2,'MarkerFaceColor',[0 0 1])
plot(angspace_ang2,ci_mem_cw,'Color',[0 .5 0],'LineWidth',.5, 'LineStyle', ':')
plot(angspace_ang2,ci_mem_ccw,'Color',[0 0 1],'LineWidth',.5, 'LineStyle', ':')
line(linspace(-0,0),linspace(-0.015,0.03001),'Color',[0 0 0])
xticks([angspace_ang2(1:2:end)])
set(gca,'TickDir','out')
ylim([-0.015 0.03001]);
xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')

subplot(1,3,2)
title('Impulse 1')
hold all
plot(angspace_ang2,mean(imp1_cw2,1),'-ro','Color',[0 .5 0],'LineWidth',2,'MarkerFaceColor',[0 .5 0])
plot(angspace_ang2,mean(imp1_ccw2,1),'-ro','Color',[0 0 1],'LineWidth',2,'MarkerFaceColor',[0 0 1])
plot(angspace_ang2,ci_imp1_cw,'Color',[0 .5 0],'LineWidth',.5, 'LineStyle', ':')
plot(angspace_ang2,ci_imp1_ccw,'Color',[0 0 1],'LineWidth',.5, 'LineStyle', ':')
line(linspace(-0,0),linspace(-0.015,0.03001),'Color',[0 0 0])
xticks([angspace_ang2(1:2:end)])
set(gca,'TickDir','out')
ylim([-0.005 0.006]);xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')

subplot(1,3,3)
title('Impulse 2')
hold all
plot(angspace_ang2,mean(imp2_cw2,1),'-ro','Color',[0 .5 0],'LineWidth',2,'MarkerFaceColor',[0 .5 0])
plot(angspace_ang2,mean(imp2_ccw2,1),'-ro','Color',[0 0 1],'LineWidth',2,'MarkerFaceColor',[0 0 1])
plot(angspace_ang2,ci_imp2_cw,'Color',[0 .5 0],'LineWidth',.5, 'LineStyle', ':')
plot(angspace_ang2,ci_imp2_ccw,'Color',[0 0 1],'LineWidth',.5, 'LineStyle', ':')
line(linspace(-0,0),linspace(-0.015,0.03001),'Color',[0 0 0])
xticks([angspace_ang2(1:2:end)])
set(gca,'TickDir','out')
ylim([-0.005 0.006]);
xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')

%%
function [dat_ccw,dat_cw,angs_ccw,angs_cw]=report_split_dat_func(dat,report_err,theta)

u_theta=unique(theta);
dat_ccw=[];
dat_cw=[];
angs_ccw=[];
angs_cw=[];
for ang=u_theta'
    report_err_temp=report_err(theta==ang,1);
    
    dat_temp=dat(theta==ang,:);
    ang_temp=theta(theta==ang,1);
    
    angs_ccw=cat(1,angs_ccw,ang_temp(report_err_temp<median(report_err_temp),1));
    angs_cw=cat(1,angs_cw,ang_temp(report_err_temp>median(report_err_temp),1));
    
    dat_ccw=cat(1,dat_ccw,dat_temp(report_err_temp<median(report_err_temp),:));
    dat_cw=cat(1,dat_cw,dat_temp(report_err_temp>median(report_err_temp),:));
end
end





