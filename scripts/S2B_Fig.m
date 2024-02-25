clear all;
close all;
clc;
%% Supplemental figure 3B, from Drifting codes (2020) Wolff et al.
cd('D:\Drifting codes\upload files') %path to main dir.
addpath(genpath(cd))

% channels of interest
test_chans = {'P7';'P5';'P3';'P1';'Pz';'P2';'P4';'P6';'P8';'PO7';'PO3';'POz';'PO4';'PO8';'O2';'O1';'Oz'};

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
        dat_temp=bsxfun(@minus,dat_temp,mean(dat_temp,3)); % take relative baseline
        dat_temp=movmean(dat_temp,span,3,'Endpoints','discard'); % downsample 
        dat_temp=dat_temp(:,:,1:span:end);
        dat_imp=reshape(dat_temp,[size(dat_temp,1),size(dat_temp,2)*size(dat_temp,3)]); %combine channel and time dimensions
        
        clear ft_imp
        %%
        onset_cond=Results(incl_i,2);
        d_acc=zeros(reps,1);
        %% decode impulse onset
        for r=1:reps % repeat decoding "reps" times
            dec_acc = mahal_nominal_kfold_b(dat_imp,onset_cond,n_folds);
            d_acc(r,1)=dec_acc;
        end
        %%
        imp_onset_acc(sub,1)=mean(d_acc,1);
    end
else
    load('S3B_Fig_results.mat')
end
%% significance testing
use_NT=1; % 1=use prepared null distributions from null deocoder, 0=make null distributions by shuffling data
if use_NT
    % make t-value
    imp_onset_acc_t=FastTtest(imp_onset_acc-.5);
    load('S3B_Fig_NULL_T.mat')
    p_acc=FastPvalue(imp_onset_acc_t,imp_onset_acc_NT,1);
else
    p_acc=GroupPermTest(imp_onset_acc-.5,perms,1,'t');
end
%% make CI using bootstrapping for plotting
ci_imp=bootci(perms,@mean,imp_onset_acc);
%%
figure
hold all
b1=boxplot([imp_onset_acc],...
    'positions',1,'Widths',0.15,'Symbol','k.');
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
set(b1(:,1),'color','k');
plot(1,mean(ci_imp,1),'o','MarkerFaceColor','k','MarkerEdgeColor','none','MarkerSize',10)
plot([1 1],ci_imp','Color','k','LineWidth',3)
plot([0 2],[0 0 ],'Color','k','LineWidth',.5,'LineStyle','--')
pbaspect([1 2 1])
ylim([.45 .6])
ylabel('Decoding accuracy (%)')
set(gca,'TickDir','out')
title('Suppl. fig. 2B')


