clear all;
close all;
clc;
%% Figure 3, top row, from Drifting codes (2020) Wolff et al.
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

reps=100; % number of repeats for random subsampling and cross-validation
span=5; % number of time-points to average over (5 = 10 ms for 500 Hz)
toi=[.1 .4001];
n_folds=8; % number of folds for cross-validation
perms=100000; % number of permutations for stats and confidence intervals
do_decoding=0; %1=run the decodng (takes long time!) or 0= load in previous output
%%
if do_decoding
    for sub=1:26 % loop over subjects
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
        %%
        % always multiply orientaitons (-90 to 90) by 2 to adhere to
        % circular geometry
        item_left=Results(:,1).*2;
        item_right=Results(:,2).*2;
        cue_cond=Results(:,3);
        cued_rad=Results(:,4).*2;
        uncued_rad=Results(:,5).*2;
        %% preallocate decoding output matrices for each item and epoch
        left_dists_mem=nan(16,ang_steps,reps); left_cos_mem=nan(ang_steps,reps);
        right_dists_mem=nan(16,ang_steps,reps); right_cos_mem=nan(ang_steps,reps);
        
        cued_left_dists_imp1=nan(16,ang_steps,reps); cued_left_cos_imp1=nan(ang_steps,reps);
        cued_right_dists_imp1=nan(16,ang_steps,reps); cued_right_cos_imp1=nan(ang_steps,reps);
        
        uncued_left_dists_imp1=nan(16,ang_steps,reps); uncued_left_cos_imp1=nan(ang_steps,reps);
        uncued_right_dists_imp1=nan(16,ang_steps,reps); uncued_right_cos_imp1=nan(ang_steps,reps);
        
        cued_left_dists_imp2=cued_left_dists_imp1; cued_left_cos_imp2=cued_left_cos_imp1;
        cued_right_dists_imp2=cued_right_dists_imp1; cued_right_cos_imp2=cued_right_cos_imp1;
        
        uncued_left_dists_imp2=uncued_left_dists_imp1; uncued_left_cos_imp2=uncued_left_cos_imp1;
        uncued_right_dists_imp2=uncued_right_dists_imp1; uncued_right_cos_imp2=uncued_right_cos_imp1;
        %%
        for a=1:ang_steps % loop through each orientation space
            
            % replace orientation values to the closes value of the current
            % orientation space
            
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
            for r=1:reps % repeat decoding "reps" times
                
                %% decoding orientation separately for each item, epoch, and location
                %% item presentation
                % left item
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_mem,bin_left_m,n_folds);
                left_dists_mem(:,a,r)=mean(distances,2);
                left_cos_mem(a,r)=mean(distance_cos,2);
                
                % right item
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_mem,bin_right_m,n_folds);
                right_dists_mem(:,a,r)=mean(distances,2);
                right_cos_mem(a,r)=mean(distance_cos,2);
                %% impulse 1
                % cued left
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp1(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1),n_folds);
                cued_left_dists_imp1(:,a,r)=mean(distances,2);
                cued_left_cos_imp1(a,r)=mean(distance_cos,2);
                
                % cued right
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp1(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2),n_folds);
                cued_right_dists_imp1(:,a,r)=mean(distances,2);
                cued_right_cos_imp1(a,r)=mean(distance_cos,2);
                
                % uncued left
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp1(cue_cond(incl_i)==2,:),bin_uncued_i(cue_cond(incl_i)==2),n_folds);
                uncued_left_dists_imp1(:,a,r)=mean(distances,2);
                uncued_left_cos_imp1(a,r)=mean(distance_cos,2);
                
                % uncued right
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp1(cue_cond(incl_i)==1,:),bin_uncued_i(cue_cond(incl_i)==1),n_folds);
                uncued_right_dists_imp1(:,a,r)=mean(distances,2);
                uncued_right_cos_imp1(a,r)=mean(distance_cos,2);
                %% impulse 2
                % cued left
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp2(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1),n_folds);
                cued_left_dists_imp2(:,a,r)=mean(distances,2);
                cued_left_cos_imp2(a,r)=mean(distance_cos,2);
                
                % cued right
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp2(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2),n_folds);
                cued_right_dists_imp2(:,a,r)=mean(distances,2);
                cued_right_cos_imp2(a,r)=mean(distance_cos,2);
                
                % uncued left
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp2(cue_cond(incl_i)==2,:),bin_uncued_i(cue_cond(incl_i)==2),n_folds);
                uncued_left_dists_imp2(:,a,r)=mean(distances,2);
                uncued_left_cos_imp2(a,r)=mean(distance_cos,2);
                
                % uncued right
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp2(cue_cond(incl_i)==1,:),bin_uncued_i(cue_cond(incl_i)==1),n_folds);
                uncued_right_dists_imp2(:,a,r)=mean(distances,2);
                uncued_right_cos_imp2(a,r)=mean(distance_cos,2);
            end
        end
        % average decoding outputs over all reps and orientation spaces
        dists_mem_both(sub,:)=mean(mean(cat(3,left_dists_mem,right_dists_mem),2),3);
        cos_mem_both(sub,1)=mean(mean(cat(2,left_cos_mem,right_cos_mem),1),2);
        
        dists_imp1_cued(sub,:)=mean(mean(cat(3,cued_left_dists_imp1,cued_right_dists_imp1),2),3);
        cos_imp1_cued(sub,1)=mean(mean(cat(2,cued_left_cos_imp1,cued_right_cos_imp1),1),2);
        
        dists_imp1_uncued(sub,:)=mean(mean(cat(3,uncued_left_dists_imp1,uncued_right_dists_imp1),2),3);
        cos_imp1_uncued(sub,1)=mean(mean(cat(2,uncued_left_cos_imp1,uncued_right_cos_imp1),1),2);
        
        dists_imp2_cued(sub,:)=mean(mean(cat(3,cued_left_dists_imp2,cued_right_dists_imp2),2),3);
        cos_imp2_cued(sub,1)=mean(mean(cat(2,cued_left_cos_imp2,cued_right_cos_imp2),1),2);
        
        dists_imp2_uncued(sub,:)=mean(mean(cat(3,uncued_left_dists_imp2,uncued_right_dists_imp2),2),3);
        cos_imp2_uncued(sub,1)=mean(mean(cat(2,uncued_left_cos_imp2,uncued_right_cos_imp2),1),2);
    end
else
    load('Fig_3_top_results')
end
%% significance testing
use_NT=1; % 1=use prepared null distributions from null deocoder, 0=make null distributions by shuffling data
if use_NT
    % make t-values
    cos_mem_both_t=FastTtest(cos_mem_both);
    cos_imp1_cued_t=FastTtest(cos_imp1_cued);
    cos_imp1_uncued_t=FastTtest(cos_imp1_uncued);
    cos_imp2_cued_t=FastTtest(cos_imp2_cued);
    cos_imp2_uncued_t=FastTtest(cos_imp2_uncued);
    cos_imp1_diff_t=FastTtest(cos_imp1_cued-cos_imp1_uncued);
    cos_imp2_diff_t=FastTtest(cos_imp2_cued-cos_imp2_uncued);
    
    load('Fig_3_top_NULL_T.mat')
    
    p_mem=FastPvalue(cos_mem_both_t,cos_mem_both_NT,1);
    p_imp1_cued=FastPvalue(cos_imp1_cued_t,cos_imp1_cued_NT,1);
    p_imp1_uncued=FastPvalue(cos_imp1_uncued_t,cos_imp1_uncued_NT,1);
    p_imp1_diff=FastPvalue(cos_imp1_diff_t,cos_imp1_diff_NT,2);
    p_imp2_cued=FastPvalue(cos_imp2_cued_t,cos_imp2_cued_NT,1);
    p_imp2_uncued=FastPvalue(cos_imp2_uncued_t,cos_imp2_uncued_NT,1);
    p_imp2_diff=FastPvalue(cos_imp2_diff_t,cos_imp2_diff_NT,2);
else
    p_mem=GroupPermTest(cos_mem_both,perms,1,'t'); % one-sided
    p_imp1_cued=GroupPermTest(cos_imp1_cued,perms,1,'t'); % one-sided
    p_imp1_uncued=GroupPermTest(cos_imp1_uncued,perms,1,'t'); % one-sided
    p_imp1_diff=GroupPermTest(cos_imp1_cued-cos_imp1_uncued,perms,2,'t'); % two-sided
    p_imp2_cued=GroupPermTest(cos_imp2_cued,perms,1,'t'); % one-sided
    p_imp2_uncued=GroupPermTest(cos_imp2_uncued,perms,1,'t'); % one-sided
    p_imp2_diff=GroupPermTest(cos_imp2_cued-cos_imp2_uncued,perms,2,'t'); % two-sided
end
%% make tunning symmetrical for plotting (-90 and 90 degrees on either side, -90 = 90)
angspace_ang2=-90:11.25:90;
dists_mem_both2=cat(2,dists_mem_both,dists_mem_both(:,1));

dists_imp1_cued2=cat(2,dists_imp1_cued,dists_imp1_cued(:,1));
dists_imp1_uncued2=cat(2,dists_imp1_uncued,dists_imp1_uncued(:,1));

dists_imp2_cued2=cat(2,dists_imp2_cued,dists_imp2_cued(:,1));
dists_imp2_uncued2=cat(2,dists_imp2_uncued,dists_imp2_uncued(:,1));
%% make CIs using bootstrapping for plotting
ci_mem_both=bootci(perms,@mean,dists_mem_both2);
ci_mem_both_cos=bootci(perms,@mean,cos_mem_both);
%% Figure 3, top, left
fhandle=figure;
subplot(1,2,1)
title('Memory array')
hold all
plot(angspace_ang2,mean(dists_mem_both2(:,:),1),'-bo','Color',[0 0 .5],'LineWidth',2,'MarkerFaceColor',[0 0 .5])
plot(angspace_ang2,ci_mem_both,'Color',[0 0 .5],'LineWidth',.5, 'LineStyle', ':')
xticks([angspace_ang2(1:2:end)])
line(linspace(0,0),linspace(-0.015, 0.0300001),'Color',[0 0 0],'LineStyle',':')
set(gca,'TickDir','out')
ylim([-0.015 0.0300001]);
xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')
set(fhandle, 'Position', [100, 100, 700, 350]);

subplot(1,2,2)
hold all
b1=boxplot([cos_mem_both],...
    'positions',1,'Widths',0.1,'Symbol','b.');
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
set(b1(:,1),'color',[0 0 .5]);
plot(1,mean(cos_mem_both,1),'o','MarkerFaceColor',[0 0 .5],'MarkerEdgeColor','none','MarkerSize',10)
plot([1 1],ci_mem_both_cos','Color',[0 0 .5],'LineWidth',3)
plot([0 2],[0 0 ],'Color','k','LineWidth',.5,'LineStyle','--')
pbaspect([1 2 1])
ylim([-.0025 .015])
ylabel('Decoding accuracy (cosine vector mean)')
set(gca,'TickDir','out')
%% make CIs using bootstrapping for plotting
ci_imp1_cued=bootci(perms,@mean,dists_imp1_cued2);
ci_imp1_uncued=bootci(perms,@mean,dists_imp1_uncued2);

ci_imp1_cued_cos=bootci(perms,@mean,cos_imp1_cued);
ci_imp1_uncued_cos=bootci(perms,@mean,cos_imp1_uncued);
%% Figure 3, top, middle
fhandle=figure;
subplot(1,2,1)
title('Impulse 1')
hold all
plot(angspace_ang2,mean(dists_imp1_cued2,1),'-bo','Color',[0 0 1],'LineWidth',2,'MarkerFaceColor',[0 0 1])
plot(angspace_ang2,mean(dists_imp1_uncued2,1),'-ro','Color',[0 0 0],'LineWidth',2,'MarkerFaceColor',[0 0 0])
plot(angspace_ang2,ci_imp1_cued,'Color',[0 0 1],'LineWidth',.5, 'LineStyle', ':')
plot(angspace_ang2,ci_imp1_uncued,'Color',[0 0 0],'LineWidth',.5, 'LineStyle', ':')
line(linspace(-90,90),linspace(-90,90),'Color',[0 0 0])
xticks([angspace_ang2(1:2:end)])
set(gca,'TickDir','out')
ylim([-0.005 0.006]);xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')
legend('cued', 'uncued')
set(fhandle, 'Position', [100, 100, 700, 350]);

subplot(1,2,2)
pos=[.95 1.05];
hold all
b1=boxplot([cos_imp1_cued,cos_imp1_uncued],...
    'positions',pos,'Widths',0.02,'Symbol','b.','Labels',{'cued','uncued'});
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
set(b1(:,1),'color','b');set(b1(:,2),'color','k');
plot(pos(1),mean(cos_imp1_cued,1),'o','MarkerFaceColor','b','MarkerEdgeColor','none','MarkerSize',10)
plot(pos(2),mean(cos_imp1_uncued,1),'o','MarkerFaceColor','k','MarkerEdgeColor','none','MarkerSize',10)
plot([pos(1) pos(1)],ci_imp1_cued_cos','Color','b','LineWidth',3)
plot([pos(2) pos(2)],ci_imp1_uncued_cos','Color','k','LineWidth',3)
plot([0 2],[0 0 ],'Color','k','LineWidth',.5,'LineStyle','--')
pbaspect([1 2 1])
ylim([-.0025 .0065])
ylabel('Decoding accuracy (cosine vector mean)')
set(gca,'TickDir','out')
%% make CIs using bootstrapping for plotting
ci_imp2_cued=bootci(perms,@mean,dists_imp2_cued2);
ci_imp2_uncued=bootci(perms,@mean,dists_imp2_uncued2);

ci_imp2_cued_cos=bootci(perms,@mean,cos_imp2_cued);
ci_imp2_uncued_cos=bootci(perms,@mean,cos_imp2_uncued);
%% Figure 3, top, right
fhandle=figure;
subplot(1,2,1)
title('Impulse 2')
hold all
plot(angspace_ang2,mean(dists_imp2_cued2,1),'-bo','Color',[0 0 1],'LineWidth',2,'MarkerFaceColor',[0 0 1])
plot(angspace_ang2,mean(dists_imp2_uncued2,1),'-ro','Color',[0 0 0],'LineWidth',2,'MarkerFaceColor',[0 0 0])
plot(angspace_ang2,ci_imp2_cued,'Color',[0 0 1],'LineWidth',.5, 'LineStyle', ':')
plot(angspace_ang2,ci_imp2_uncued,'Color',[0 0 0],'LineWidth',.5, 'LineStyle', ':')
line(linspace(-90,90),linspace(-90,90),'Color',[0 0 0])
xticks([angspace_ang2(1:2:end)])
set(gca,'TickDir','out')
ylim([-0.005 0.006]);xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')
legend('cued', 'uncued')
set(fhandle, 'Position', [100, 100, 700, 350]);

subplot(1,2,2)
pos=[.95 1.05];
hold all
b1=boxplot([cos_imp2_cued,cos_imp2_uncued],...
    'positions',pos,'Widths',0.02,'Symbol','b.','Labels',{'cued','uncued'});
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
set(b1(:,1),'color','b');set(b1(:,2),'color','k');
plot(pos(1),mean(cos_imp2_cued,1),'o','MarkerFaceColor','b','MarkerEdgeColor','none','MarkerSize',10)
plot(pos(2),mean(cos_imp2_uncued,1),'o','MarkerFaceColor','k','MarkerEdgeColor','none','MarkerSize',10)
plot([pos(1) pos(1)],ci_imp2_cued_cos','Color','b','LineWidth',3)
plot([pos(2) pos(2)],ci_imp2_uncued_cos','Color','k','LineWidth',3)
plot([0 2],[0 0 ],'Color','k','LineWidth',.5,'LineStyle','--')
pbaspect([1 2 1])
ylim([-.0025 .0065])
ylabel('Decoding accuracy (cosine vector mean)')
set(gca,'TickDir','out')