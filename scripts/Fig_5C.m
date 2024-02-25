clear all;
close all;
clc;
%% Figure 5C, from Drifting codes (2020) Wolff et al.
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
        
        dat_imps=(dat_imp1+dat_imp2)./2; % combine impulses
        %%
        cue_cond=Results(:,3);
        cued_rad=Results(:,4).*2;
        %% preallocate decoding output matrices for each item and epoch
        cued_left_dists_same=nan(16,ang_steps,reps); cued_left_cos_same=nan(ang_steps,reps);
        cued_right_dists_same=nan(16,ang_steps,reps); cued_right_cos_same=nan(ang_steps,reps);
        
        cued_left_dists_diff=cued_left_dists_same; cued_left_cos_diff=cued_left_cos_same;
        cued_right_dists_diff=cued_right_dists_same; cued_right_cos_diff=cued_right_cos_same;
        %%
        for a=1:ang_steps % loop through each orientation space
            
            % replace orientation values to the closes value of the current
            % orientation space

            ang_bin_temp=repmat(angspaces(:,a),[1 length(cued_rad)]);
            [~,ind]= min(abs(circ_dist2(angspaces(:,a),cued_rad)));
            bin_cued=ang_bin_temp(ind)';
            
            bin_cued_i=bin_cued(incl_i,1);
            %%
            for r=1:reps % repeat decoding "reps" times
                %% train on same side
                % cued left
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imps(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1),n_folds);
                cued_left_dists_same(:,a,r)=mean(distances,2);
                cued_left_cos_same(a,:,r)=mean(distance_cos,2);
                
                % cued right
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imps(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2),n_folds);
                cued_right_dists_same(:,a,r)=mean(distances,2);
                cued_right_cos_same(a,:,r)=mean(distance_cos,2);
                %% train on different side
                % cued left
                [distance_cos,distances] = mahal_theta_basis_b_ind(dat_imps(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1),...
                    dat_imps(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2));
                cued_left_dists_diff(:,a,r)=mean(distances,2);
                cued_left_cos_diff(a,:,r)=mean(distance_cos,2);
                
                % cued right
                [distance_cos,distances] = mahal_theta_basis_b_ind(dat_imps(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2),...
                    dat_imps(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1));
                cued_right_dists_diff(:,a,r)=mean(distances,2);
                cued_right_cos_diff(a,:,r)=mean(distance_cos,2);
            end
        end
        
        % average into same and different side training results
        dists_imps_cued_same(sub,:)=mean(mean(mean(cat(3,cued_left_dists_same,cued_right_dists_same),2),3),4);
        cos_imps_cued_same(sub,1)=mean(mean(mean(cat(2,cued_left_cos_same,cued_right_cos_same),1),2),3);
        
        dists_imps_cued_diff(sub,:)=mean(mean(mean(cat(3,cued_left_dists_diff,cued_right_dists_diff),2),3),4);
        cos_imps_cued_diff(sub,1)=mean(mean(mean(cat(2,cued_left_cos_diff,cued_right_cos_diff),1),2),3);
        
    end
else
    load('Fig_5C_results')
end
%% significance testing
use_NT=1; % 1=use prepared null distributions from null deocoder, 0=make null distributions by shuffling data
if use_NT
    cos_same_t=FastTtest(cos_imps_cued_same);
    cos_different_t=FastTtest(cos_imps_cued_diff);
    cos_diffs_t=FastTtest(cos_imps_cued_same-cos_imps_cued_diff);
    
    load('Fig_5C_NULL_T.mat');
    
    p_same=FastPvalue(cos_same_t,cos_imp_same_NT,1); % one-sided
    p_different=FastPvalue(cos_different_t,cos_imp_different_NT,1); % one-sided
    p_diff=FastPvalue(cos_diffs_t,cos_imp_diffs_NT,2);
else
    p_same=GroupPermTest(cos_imps_cued_same,perms,1,'t'); % one-sided
    p_different=GroupPermTest(cos_imps_cued_diff,perms,1,'t'); % one-sided
    p_diff=GroupPermTest(cos_imps_cued_same-cos_imps_cued_diff,perms,2,'t'); % two-sided
end

%% make tunning symmetrical for plotting (-90 and 90 degrees on either side, -90 = 90)
angspace_ang2=-90:11.25:90;

dists_same2=cat(2,dists_imps_cued_same,dists_imps_cued_same(:,1));
dists_different2=cat(2,dists_imps_cued_diff,dists_imps_cued_diff(:,1));
%% make CIs using bootstrapping for plotting
ci_dists_s=bootci(sims,@mean,dists_same2);
ci_dists_d=bootci(sims,@mean,dists_different2);

ci_cos_s=bootci(sims,@mean,cos_imps_cued_same);
ci_cos_d=bootci(sims,@mean,cos_imps_cued_diff);
%% Figure 5C
fhandle=figure;
subplot(1,2,1)
hold all
plot(angspace_ang2,mean(dists_same2,1),'-ro','Color',[0 0 1],'LineWidth',2,'MarkerFaceColor',[0 0 1])
plot(angspace_ang2,mean(dists_different2,1),'-ro','Color',[0 .5 0],'LineWidth',2,'MarkerFaceColor',[0 .5 0])
plot(angspace_ang2,ci_dists_s,'Color',[0 0 1],'LineWidth',.5, 'LineStyle', ':')
plot(angspace_ang2,ci_dists_d,'Color',[0 .5 0],'LineWidth',.5, 'LineStyle', ':')
line(linspace(0,0),linspace(-0.005, 0.006),'Color',[0 0 0],'LineStyle',':')
xticks([angspace_ang2(1:2:end)])
set(gca,'TickDir','out')
ylim([-0.005 0.006]);xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')
legend('same', 'different')
set(fhandle, 'Position', [100, 100, 700, 350]);

subplot(1,2,2)
pos=[.95 1.05];
hold all
b1=boxplot([cos_imps_cued_same,cos_imps_cued_diff],...
    'positions',pos,'Widths',0.02,'Symbol','b.','Labels',{'same','different'});
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
set(b1(:,1),'color','b');set(b1(:,2),'color','k');
plot(pos(1),mean(cos_imps_cued_same,1),'o','MarkerFaceColor','b','MarkerEdgeColor','none','MarkerSize',10)
plot(pos(2),mean(cos_imps_cued_diff,1),'o','MarkerFaceColor',[0 .5 0],'MarkerEdgeColor','none','MarkerSize',10)
plot([pos(1) pos(1)],ci_cos_s','Color','b','LineWidth',3)
plot([pos(2) pos(2)],ci_cos_d','Color',[0 .5 0],'LineWidth',3)
plot([0 2],[0 0 ],'Color','k','LineWidth',.5,'LineStyle','--')
pbaspect([1 2 1])
ylim([-.0025 .0065])
ylabel('Decoding accuracy (cosine vector mean)')
set(gca,'TickDir','out')
