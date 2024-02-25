close all
clear all
clc
%% Figure 2B & Supplemental fig. 1, from Drifting codes (2020) Wolff et al.
cd('D:\Drifting codes\upload files') %path to main dir.
addpath(genpath(cd))

load('Behav_results_all.mat')
%% Figure 2B
report_err_all=Results_all(:,:,6);
report_err_all=report_err_all(:);
%
figure
hold all 
histogram(report_err_all,[linspace(-90, 90,37)],'FaceColor' ,[0.3 0.3 0.3],'FaceAlpha',1)
xlim([-90 90])
ylim([0 5500])
set(gca,'TickDir','out');
xticks([-90:30:90])
pbaspect([1 1 1])
ylabel('Number of trials (all subjects)')
xlabel('Report error')
%% Supplemental fig. 1
bin_width=pi/8;
ang_steps=8;
for sub=1:26
    % remove trials based on bad performance (>circular SDs)
    report_err=(circ_ang2rad(Results_all(sub,:,6).*2))';
    [~,SD]=circ_std(report_err);
    incl_ng=find(abs(circ_dist(report_err,circ_mean(report_err)))<(3*SD));
    report_ng=report_err(incl_ng,1);
    report_err_ng_all{sub}=circ_rad2ang(report_ng./2);
    
    % remove response bias by subtracting the median response error within
    % each angluar bin
    cued_rad_ng=Results_all(sub,incl_ng,4).*2;
    angspace_temp=(-pi:bin_width:pi)';
    angspace_temp(end)=[];
    report_err_adj=nan(length(cued_rad_ng),ang_steps);
    for a=1:ang_steps
        angspace=angspace_temp+(a-1)*bin_width/ang_steps;
        [~,cued_cond]= min(abs(circ_dist2(angspace,cued_rad_ng)));
        for c=1:length(angspace)
            inds=cued_cond==c;
            mean_temp=circ_median(report_ng(cued_cond==c,1));
            report_err_adj(inds,a)=circ_dist(report_ng(cued_cond==c,1),mean_temp);
        end
    end
    report_err_adj_all{sub}=circ_rad2ang(mean(report_err_adj,2)./2);
    presented_ng_all{sub}=circ_rad2ang(cued_rad_ng)'./2;
end
%% 
% (left)
figure
histogram2(cat(1,presented_ng_all{:}),cat(1,report_err_ng_all{:}),linspace(-90, 90,37),linspace(-90, 90,37),...
    'DisplayStyle','tile','ShowEmptyBins','on','LineStyle','none')
xlim([-90 90])
xticks([-90:30:90])
ylim([-60 60])
yticks([-90:30:90])
pbaspect([1.5 1 1])
xlabel('presented orientation')
ylabel('report error')
set(gca,'TickDir','out');
colorbar
colormap('magma')

% (middle)
figure
histogram2(cat(1,presented_ng_all{:}),cat(1,report_err_adj_all{:}),linspace(-90, 90,37),linspace(-90, 90,37),...
    'DisplayStyle','tile','ShowEmptyBins','on','LineStyle','none')
xlim([-90 90])
xticks([-90:30:90])
ylim([-60 60])
yticks([-90:30:90])
pbaspect([1.5 1 1])
xlabel('presented orientation')
ylabel('adjusted report error')
set(gca,'TickDir','out');
colorbar
colormap('magma')

% (right)
figure
hold all
histogram(cat(1,report_err_ng_all{:}),[linspace(-90, 90,37)],'FaceColor' ,[0.3 0.3 0.3],'FaceAlpha',1)
histogram(cat(1,report_err_adj_all{:}),[linspace(-90, 90,37)],'FaceColor' ,[0 .8 .8],'FaceAlpha',.5)
xlim([-90 90])
ylim([0 5500])
set(gca,'TickDir','out');
xticks([-90:30:90])
pbaspect([1 1 1])
legend('with report bias','without report bias')
ylabel('Number of trials (all subjects)')
xlabel('(Adjusted) report error')
