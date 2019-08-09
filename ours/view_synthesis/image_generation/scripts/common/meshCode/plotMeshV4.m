function h = plotMeshV4(vertices,faces,points)

    if nargin==5

    else
        num_vertices = size(vertices,1);
%        color = repmat([245/255 245/255 245/255],num_vertices,1);
        color = repmat([187/255 1 1],num_vertices,1);
    end
    vv = color;
    h = figure('color','w','renderer','zbuffer','units','pixels','position',[100 100 600 600]);

    patch('Vertices',vertices,'Faces',faces,'FaceVertexCData',color,'FaceColor','flat');
    axis image off;
    shading flat
    hold on;

    sphere = TriangulatedSphere(50);
    for g = 1:size(points,1)
        axisPoint1 = points(g,:);
        sphere1 = sphere;
        sphere1.X = 5*sphere.X + axisPoint1(1,1);
        sphere1.Y = 5*sphere.Y + axisPoint1(1,2);
        sphere1.Z = 5*sphere.Z + axisPoint1(1,3);
        trisurf(sphere1.TRI,sphere1.X,sphere1.Y,sphere1.Z,'facecolor',[245/255 245/255 245/255],'edgecolor','none');
        hold on;
    end
    set(gcf,'Renderer','OpenGL');
    alpha(1); axis on; axis image; axis tight; zoom(1); axis off
    lighting phong; %phong; gouraud; flat;
    light('Position',[-2 2 2],'Style','local'); material shiny;
    material([.5 .5 0.2 25 0]); cameramenu;
%      light
%      camlight head
%      lighting phong	
    
end


