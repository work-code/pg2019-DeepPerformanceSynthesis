function h = plotPoints2D(vertices,color)

%h = scatter(vertices(:,1),vertices(:,2),10,color,'filled');
h = figure(234);clf; set(gcf,'color','w','position',[100 100 500 300])
for g = 1:size(vertices,1)
    plot(vertices(g,1),vertices(g,2),'.','color',color(g,:),'MarkerSize',40);
    hold on;
end
    axis image off;
    shading flat
    hold on;

end