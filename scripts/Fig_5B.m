clear all;
close all;
clc;
%% Figure 5B from Drifting codes (2020) Wolff et al.
 cd('D:\Drifting codes\upload files') %path to main dir.
addpath(genpath(cd))

% channels of interest
test_chans = {'P7';'P5';'P3';'P1';'Pz';'P2';'P4';'P6';'P8';'PO7';'PO3';'POz';'PO4';'PO8';'O2';'O1';'Oz'};

reps=100;  % number of repeats for random subsampling and cross-validation
span=5; % number of time-points to average over (5 = 10 ms for 500 Hz)
toi=[.1 .4001];
n_folds=8; % number of folds for cross-validation
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
        d_acc=zeros(reps,1);
        %% decode cued side
        for r=1:reps % repeat decoding "reps" times
            dec_acc = mahal_nominal_kfold_b(dat_imps,cue_cond,n_folds);
            d_acc(r,1)=dec_acc;
        end
        %%
        imps_side_acc(sub,1)=mean(d_acc,1);
    end
else
    load('Fig_5B_results.mat')
end
%% significance testing
use_NT=1; % 1=use prepared null distributions from null deocoder, 0=make null distributions by shuffling data
if use_NT
    % make t-value
    imps_side_acc_t=FastTtest(imps_side_acc-0.5);
    load('Fig_5B_NULL_T.mat')
    p_acc=FastPvalue(imps_side_acc_t,imps_side_acc_NT,1);
else
    p_acc=GroupPermTest(imps_side_acc-.5,perms,1,'t');
end
%% make CI using bootstrapping for plotting
ci_imp=bootci(perms,@mean,imps_side_acc);
%%
figure
hold all
b1=boxplot([imps_side_acc],...
    'positions',1,'Widths',0.15,'Symbol','k.');
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
set(b1(:,1),'color','k');
plot(1,mean(ci_imp,1),'o','MarkerFaceColor','k','MarkerEdgeColor','none','MarkerSize',10)
plot([1 1],ci_imp','Color','k','LineWidth',3)
plot([0 2],[0 0 ],'Color','k','LineWidth',.5,'LineStyle','--')
pbaspect([1 2 1])
ylim([.5 1])
ylabel('Decoding accuracy (%)')
set(gca,'TickDir','out')
title('Fig. 5B')


