function plotrate
p.roc = (1:1:205);

filename = 'F:\4_researchWork3\2_report\20180326\memo\figure\png\';
desc =1;
PlotAccRate2(accRate,p,filename,desc);
end

function SavePlot(f,fileName)
t = tempname;
saveas(f, t, 'png');  movefile([t '.png'],[fileName '.png']); 
saveas(f, t, 'fig');  movefile([t '.fig'],[fileName '.fig']); 
saveas(f, t, 'epsc'); movefile([t '.eps'],[fileName '.eps']); 
end

function PlotAccRate2(accRate,p,filename,desc)
clrs = {'b','g','b','r'};
style = {'-','--','--','-'};
linewidth = {2,2,2,2};
f  = figure('visible','off');
set(f,'position',[100 100 1000 600]);
num = 1;


for k = 1:num
    acc = accRate;
    tmp.accRate = acc;
    tmp.geoerr = p.roc;

    h = AddToPlots2(f,tmp,style{k},clrs{k},linewidth{k});
end


figure(f);
%xlabel('false positive rate')
%ylabel('true positive rate')

%legend('HKS','WKS','OSD','Ours');
%legend('HKS','WKS');
%title(p.dataset)


SavePlot(f,filename)
end


function h = AddToPlots2(f,data,styl,clr,linewidth)
%linewidth = 4;
 
cmcX = data.geoerr;
cmcY = data.accRate;
% % values = spcrv([[cmcX(1) cmcX cmcX(end)];[cmcY(1) cmcY cmcY(end)]],3);
% xq=linspace(min(cmcX),max(cmcX));
% yq=interp1(cmcX,cmcY,xq,'spline');
% set(0,'currentFigure',f);

h = plot(cmcX,cmcY,styl,'color',clr,'linewidth',linewidth);% return
axis([0,205,0,0.15])
set(gca,'XTick',0:15:205);
set(gca,'DataAspectRatio',[2500,3.2,1]);
hold on;
%grid on;
box on;
end