function h = plotCores(vertices,color)
    S = 50*ones(1,size(vertices,1));
    if nargin==2

    else
        num_vertices = size(vertices,1);
        color = repmat([187/255 1 1],num_vertices,1);
    end
    vv = color;
    %color = (vv-min(vv))./(max(vv)-min(vv)+eps);
    
    h = figure(234);clf; set(gcf,'color','w','position',[100 100 500 500])
    hold on;
    view(3);
    set(gca,'position',[0 0 1 1]);
    sphere = TriangulatedSphere(100);
    for i = 1:size(vertices,1)
        axisPoint1 = vertices(i,:);
        sphere1 = sphere;
        sphere1.X = 2*sphere.X + axisPoint1(1);
        sphere1.Y = 2*sphere.Y + axisPoint1(2);
        sphere1.Z = 2*sphere.Z + axisPoint1(3);
        trisurf(sphere1.TRI,sphere1.X,sphere1.Y,sphere1.Z,'facecolor',color(i,:),'edgecolor','none')
    end 
    axis image off;
    shading flat;
    light;
    camlight head;
    lighting phong;
    	
end