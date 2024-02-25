clear all;
close all;
clc;
%% Supplemental figure 3C, from Drifting codes (2020) Wolff et al.
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
    for sub=1:24
        fprintf(['Doing ' num2str(sub) '\n'])
        
        load(['dat_2015_' num2str(sub) '.mat']);
        
        % trials to include (based on performance and previously marked
        % "bad" trials
        incl_i=setdiff(1:size(Results,1), ft_imp.bad_trials);
        
        % extract good trials, channels and time window of interest of
        % impulse epoch and reformat
        dat_temp=ft_imp.trial(incl_i,ismember(ft_imp.label,test_chans),ft_imp.time>toi(1)&ft_imp.time<=toi(2));
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3));
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard');
        dat_temp=dat_temp(:,:,1:span:end);
        dat_imp=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]);
        
        clear ft_imp        
        %%
        onset_cond=Results(incl_i,2);
        item_rad=Results(incl_i,1).*2;
        %% preallocate decoding output matrices
        item_early_dists_same=nan(16,ang_steps,reps); item_early_cos_same=nan(ang_steps,reps);
        item_late_dists_same=item_early_dists_same; item_late_cos_same=item_early_cos_same;
        
        item_early_dists_diff=item_early_dists_same; item_early_cos_diff=item_early_cos_same;
        item_late_dists_diff=item_early_dists_same; item_late_cos_diff=item_early_cos_same;
        %%
        for a=1:ang_steps % loop through each orientation space
            
            % replace orientation values to the closes value of the current
            % orientation space
            
            ang_bin_temp=repmat(angspaces(:,a),[1 length(item_rad)]);
            [~,ind]= min(abs(circ_dist2(angspaces(:,a),item_rad)));
            bin_item=ang_bin_temp(ind)';
            %%
            for r=1:reps % repeat decoding "reps" times
                %% train on same onset
                % early
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp(onset_cond==1,:),bin_item(onset_cond==1),n_folds);
                item_early_dists_same(:,a,r)=mean(distances,2);
                item_early_cos_same(a,r)=mean(distance_cos,2);
                
                % late
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp(onset_cond==2,:),bin_item(onset_cond==2),n_folds);
                item_late_dists_same(:,a,r)=mean(distances,2);
                item_late_cos_same(a,r)=mean(distance_cos,2);
                %% train on different onset
                % early
                [distance_cos,distances] = mahal_theta_basis_b_ind(dat_imp(onset_cond==1,:),bin_item(onset_cond==1),...
                    dat_imp(onset_cond==2,:),bin_item(onset_cond==2));
                item_early_dists_diff(:,a,r)=mean(distances,2);
                item_early_cos_diff(a,r)=mean(distance_cos,2);
                
                % late
                [distance_cos,distances] = mahal_theta_basis_b_ind(dat_imp(onset_cond==2,:),bin_item(onset_cond==2),...
                    dat_imp(onset_cond==1,:),bin_item(onset_cond==1));
                item_late_dists_diff(:,a,r)=mean(distances,2);
                item_late_cos_diff(a,r)=mean(distance_cos,2);
            end
        end
        
        % average into same and different onset training results
        dists_imps_same(sub,:)=mean(mean(cat(3,item_early_dists_same,item_late_dists_same),2),3);
        cos_imps_same(sub,1)=mean(mean(cat(2,item_early_cos_same,item_late_cos_same),1),2);
        
        dists_imps_diff(sub,:)=mean(mean(cat(3,item_early_dists_diff,item_late_dists_diff),2),3);
        cos_imps_diff(sub,1)=mean(mean(cat(2,item_early_cos_diff,item_late_cos_diff),1),2);
    end
else
    load('S3C_Fig_results.mat')
end
%% significance testing
use_NT=1; % 1=use prepared null distributions from null deocoder, 0=make null distributions by shuffling data
if use_NT
    cos_same_t=FastTtest(cos_imps_same);
    cos_different_t=FastTtest(cos_imps_diff);
    cos_diffs_t=FastTtest(cos_imps_same-cos_imps_diff);
    
    load('S3C_Fig_NULL_T.mat');
    
    p_same=FastPvalue(cos_same_t,cos_imp_same_NT,1); % one-sided
    p_different=FastPvalue(cos_different_t,cos_imp_diff_NT,1); % one-sided
    p_diff=FastPvalue(cos_diffs_t,cos_imp_diffs_NT,2);
else
    p_same=GroupPermTest(cos_imps_same,perms,1,'t'); % one-sided
    p_different=GroupPermTest(cos_imps_diff,perms,1,'t'); % one-sided
    p_diff=GroupPermTest(cos_same-cos_different,perms,2,'t'); % two-sided
end
%% make tunning symmetrical for plotting (-90 and 90 degrees on either side, -90 = 90)
angspace_ang2=-90:11.25:90;

dists_same2=cat(2,dists_imps_same,dists_imps_same(:,1));
dists_different2=cat(2,dists_imps_diff,dists_imps_diff(:,1));
%% make CIs using bootstrapping for plotting
ci_dists_s=bootci(perms,@mean,dists_same2);
ci_dists_d=bootci(perms,@mean,dists_different2);

ci_cos_s=bootci(perms,@mean,cos_imps_same);
ci_cos_d=bootci(perms,@mean,cos_imps_diff);
%% Suppl. Fig. 3C
fhandle=figure;
subplot(1,2,1)
hold all
plot(angspace_ang2,mean(dists_same2,1),'-ro','Color',[0 0 1],'LineWidth',2,'MarkerFaceColor',[0 0 1])
plot(angspace_ang2,mean(dists_different2,1),'-ro','Color',[0 .5 0],'LineWidth',2,'MarkerFaceColor',[0 .5 0])
plot(angspace_ang2,ci_dists_s,'Color',[0 0 1],'LineWidth',.5, 'LineStyle', ':')
plot(angspace_ang2,ci_dists_d,'Color',[0 .5 0],'LineWidth',.5, 'LineStyle', ':')
line(linspace(0,0),linspace(-0.007, 0.012),'Color',[0 0 0],'LineStyle',':')
xticks([angspace_ang2(1:2:end)])
set(gca,'TickDir','out')
ylim([-0.007 0.012]);xlim([-95 95])
xlabel('Orientation difference (degrees)')
ylabel('Pattern similarity (norm. volt.)')
pbaspect([1,1,1])
set(gca,'TickDir','out')
legend('same', 'different')
set(fhandle, 'Position', [100, 100, 700, 350]);

subplot(1,2,2)
pos=[.95 1.05];
hold all
b1=boxplot([cos_imps_same,cos_imps_diff],...
    'positions',pos,'Widths',0.02,'Symbol','b.','Labels',{'same','different'});
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
set(b1(:,1),'color','b');set(b1(:,2),'color','k');
plot(pos(1),mean(cos_imps_same,1),'o','MarkerFaceColor','b','MarkerEdgeColor','none','MarkerSize',10)
plot(pos(2),mean(cos_imps_diff,1),'o','MarkerFaceColor',[0 .5 0],'MarkerEdgeColor','none','MarkerSize',10)
plot([pos(1) pos(1)],ci_cos_s','Color','b','LineWidth',3)
plot([pos(2) pos(2)],ci_cos_d','Color',[0 .5 0],'LineWidth',3)
plot([0 2],[0 0 ],'Color','k','LineWidth',.5,'LineStyle','--')
pbaspect([1 2 1])
ylim([-.001 .01])
ylabel('Decoding accuracy (cosine vector mean)')
set(gca,'TickDir','out')
