clear all;
close all;
clc;
%% Figure 4C, from Drifting codes (2020) Wolff et al.
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
        %%
        cue_cond=Results(:,3);
        
        % always multiply orientaitons (-90 to 90) by 2 to adhere to
        % circular geometry
        cued_rad=Results(:,4).*2;
        %% preallocate decoding output matrices for each item and epoch
        cued_left_dists_imp1=nan(16,ang_steps,reps); cued_left_cos_imp1=nan(ang_steps,reps);
        cued_right_dists_imp1=nan(16,ang_steps,reps); cued_right_cos_imp1=nan(ang_steps,reps);
        
        cued_left_dists_imp2=cued_left_dists_imp1; cued_left_cos_imp2=cued_left_cos_imp1;
        cued_right_dists_imp2=cued_right_dists_imp1; cued_right_cos_imp2=cued_right_cos_imp1;
        
        cued_left_dists_imp2_trn1=cued_left_dists_imp1; cued_left_cos_imp2_trn1=cued_left_cos_imp1;
        cued_right_dists_imp2_trn1=cued_right_dists_imp1; cued_right_cos_imp2_trn1=cued_right_cos_imp1;
        
        cued_left_dists_imp1_trn2=cued_left_dists_imp1; cued_left_cos_imp1_trn2=cued_left_cos_imp1;
        cued_right_dists_imp1_trn2=cued_right_dists_imp1; cued_right_cos_imp1_trn2=cued_right_cos_imp1;
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
                %% impulse 1
                % cued left
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp1(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1),n_folds);
                cued_left_dists_imp1(:,a,r)=mean(distances,2);
                cued_left_cos_imp1(a,r)=mean(distance_cos,2);
                
                % cued right
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp1(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2),n_folds);
                cued_right_dists_imp1(:,a,r)=mean(distances,2);
                cued_right_cos_imp1(a,r)=mean(distance_cos,2);
                %% impulse 2
                % cued left
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp2(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1),n_folds);
                cued_left_dists_imp2(:,a,r)=mean(distances,2);
                cued_left_cos_imp2(a,r)=mean(distance_cos,2);
                
                % cued right
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp2(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2),n_folds);
                cued_right_dists_imp2(:,a,r)=mean(distances,2);
                cued_right_cos_imp2(a,r)=mean(distance_cos,2);
                %% impulse 1, train on impulse 2
                % cued left
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp1(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1),n_folds,dat_imp2(cue_cond(incl_i)==1,:));
                cued_left_dists_imp1_trn2(:,a,r)=mean(distances,2);
                cued_left_cos_imp1_trn2(a,r)=mean(distance_cos,2);
                
                % cued right
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp1(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2),n_folds,dat_imp2(cue_cond(incl_i)==2,:));
                cued_right_dists_imp1_trn2(:,a,r)=mean(distances,2);
                cued_right_cos_imp1_trn2(a,r)=mean(distance_cos,2);
                %% impulse 2, train on impulse 1
                % cued left
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp2(cue_cond(incl_i)==1,:),bin_cued_i(cue_cond(incl_i)==1),n_folds,dat_imp1(cue_cond(incl_i)==1,:));
                cued_left_dists_imp2_trn1(:,a,r)=mean(distances,2);
                cued_left_cos_imp2_trn1(a,r)=mean(distance_cos,2);
                
                % cued right
                [distance_cos,distances] = mahal_theta_kfold_basis_b(dat_imp2(cue_cond(incl_i)==2,:),bin_cued_i(cue_cond(incl_i)==2),n_folds,dat_imp1(cue_cond(incl_i)==2,:));
                cued_right_dists_imp2_trn1(:,a,r)=mean(distances,2);
                cued_right_cos_imp2_trn1(a,r)=mean(distance_cos,2);
            end
        end
        dists_imp1_cued(sub,:)=mean(mean(cat(3,cued_left_dists_imp1,cued_right_dists_imp1),2),3);
        cos_imp1_cued(sub,1)=mean(mean(cat(2,cued_left_cos_imp1,cued_right_cos_imp1),1),2);
        
        dists_imp2_cued(sub,:)=mean(mean(cat(3,cued_left_dists_imp2,cued_right_dists_imp2),2),3);
        cos_imp2_cued(sub,1)=mean(mean(cat(2,cued_left_cos_imp2,cued_right_cos_imp2),1),2);
        
        dists_imp1_cued_trn2(sub,:)=mean(mean(cat(3,cued_left_dists_imp1_trn2,cued_right_dists_imp1_trn2),2),3);
        cos_imp1_cued_trn2(sub,1)=mean(mean(cat(2,cued_left_cos_imp1_trn2,cued_right_cos_imp1_trn2),1),2);
        
        dists_imp2_cued_trn1(sub,:)=mean(mean(cat(3,cued_left_dists_imp2_trn1,cued_right_dists_imp2_trn1),2),3);
        cos_imp2_cued_trn1(sub,1)=mean(mean(cat(2,cued_left_cos_imp2_trn1,cued_right_cos_imp2_trn1),1),2);
    end
    
    % average into same and different epoch training results
    dists_same=(dists_imp1_cued+dists_imp2_cued)./2;
    dists_different=(dists_imp1_cued_trn2+dists_imp2_cued_trn1)./2;
    
    cos_same=(cos_imp1_cued+cos_imp2_cued)./2;
    cos_different=(cos_imp1_cued_trn2+cos_imp2_cued_trn1)./2;
else
    load('Fig_4C_results')
end
%% significance testing
use_NT=1; % 1=use prepared null distributions from null deocoder, 0=make null distributions by shuffling data
if use_NT
    cos_same_t=FastTtest(cos_same);
    cos_different_t=FastTtest(cos_different);
    cos_diffs_t=FastTtest(cos_same-cos_different);
    
    load('Fig_4C_NULL_T.mat');
    
    p_same=FastPvalue(cos_same_t,cos_imp_same_NT,1); % one-sided
    p_different=FastPvalue(cos_different_t,cos_imp_diff_NT,1); % one-sided
    p_diff=FastPvalue(cos_diffs_t,cos_imp_diffs_NT,2);
else
    p_same=GroupPermTest(cos_same,perms,1,'t'); % one-sided
    p_different=GroupPermTest(cos_different,perms,1,'t'); % one-sided
    p_diff=GroupPermTest(cos_same-cos_different,perms,2,'t'); % two-sided
end
%% make tunning symmetrical for plotting (-90 and 90 degrees on either side, -90 = 90)
angspace_ang2=-90:11.25:90;

dists_same2=cat(2,dists_same,dists_same(:,1));
dists_different2=cat(2,dists_different,dists_different(:,1));
%% make CIs using bootstrapping for plotting
ci_dists_s=bootci(perms,@mean,dists_same2);
ci_distsd=bootci(perms,@mean,dists_different2);

ci_cos_w=bootci(perms,@mean,cos_same);
ci_cos_b=bootci(perms,@mean,cos_different);
%% Figure 4C
fhandle=figure;
subplot(1,2,1)
hold all
plot(angspace_ang2,mean(dists_same2,1),'-ro','Color',[0 0 1],'LineWidth',2,'MarkerFaceColor',[0 0 1])
plot(angspace_ang2,mean(dists_different2,1),'-ro','Color',[0 .5 0],'LineWidth',2,'MarkerFaceColor',[0 .5 0])
plot(angspace_ang2,ci_dists_s,'Color',[0 0 1],'LineWidth',.5, 'LineStyle', ':')
plot(angspace_ang2,ci_distsd,'Color',[0 .5 0],'LineWidth',.5, 'LineStyle', ':')
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
b1=boxplot([cos_same,cos_different],...
    'positions',pos,'Widths',0.02,'Symbol','b.','Labels',{'same','different'});
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
set(b1(:,1),'color','b');set(b1(:,2),'color','k');
plot(pos(1),mean(cos_same,1),'o','MarkerFaceColor','b','MarkerEdgeColor','none','MarkerSize',10)
plot(pos(2),mean(cos_different,1),'o','MarkerFaceColor',[0 .5 0],'MarkerEdgeColor','none','MarkerSize',10)
plot([pos(1) pos(1)],ci_cos_w','Color','b','LineWidth',3)
plot([pos(2) pos(2)],ci_cos_b','Color',[0 .5 0],'LineWidth',3)
plot([0 2],[0 0 ],'Color','k','LineWidth',.5,'LineStyle','--')
pbaspect([1 2 1])
ylim([-.0025 .0065])
ylabel('Decoding accuracy (cosine vector mean)')
set(gca,'TickDir','out')
