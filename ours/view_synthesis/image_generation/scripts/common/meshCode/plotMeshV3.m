function h = plotMeshV3(vertices,faces,pVertices,color2,color)

    if nargin==5

    else
        num_vertices = size(vertices,1);
%        color = ones(num_vertices,1);
        %color = repmat([187/255 1 1],num_vertices,1);
        %color = repmat([245/255 245/255 245/255],num_vertices,1);
        color = repmat([245/255 245/255 245/255],num_vertices,1);
    end
    vv = color;
    %color = (vv-min(vv))./(max(vv)-min(vv)+eps);
    h = figure('color','w','renderer','zbuffer','units','pixels','position',[100 100 600 600]);

    
    
    patch('Vertices',vertices,'Faces',faces,'FaceVertexCData',color,'FaceColor','flat');
    axis image off;
    shading flat
    hold on;

    sphere = TriangulatedSphere(50);
    for i = 1:size(pVertices,1)
        axisPoint1 = vertices(pVertices(i,:),:);
        sphere1 = sphere;
        sphere1.X = 1.5*sphere.X + axisPoint1(1,1);
        sphere1.Y = 1.5*sphere.Y + axisPoint1(1,2);
        sphere1.Z = 1.5*sphere.Z + axisPoint1(1,3);
%        trisurf(sphere1.TRI,sphere1.X,sphere1.Y,sphere1.Z,'facecolor',color2(i,:),'edgecolor','none');
        trisurf(sphere1.TRI,sphere1.X,sphere1.Y,sphere1.Z,'facecolor',[0,206,209]./255,'edgecolor','none');
        hold on;
    end 
    axisPoint1 = vertices(4688,:);
    sphere1 = sphere;
    sphere1.X = 3*sphere.X + axisPoint1(1,1);
    sphere1.Y = 3*sphere.Y + axisPoint1(1,2);
    sphere1.Z = 3*sphere.Z + axisPoint1(1,3);
    trisurf(sphere1.TRI,sphere1.X,sphere1.Y,sphere1.Z,'facecolor',[255,99,71]./255,'edgecolor','none');
    hold on;
    set(gcf,'Renderer','OpenGL');
    alpha(1); axis on; axis image; axis tight; zoom(1); axis off
    lighting phong; %phong; gouraud; flat;
    light('Position',[-2 2 2],'Style','local'); material shiny;
    material([.5 .5 0.2 25 0]); cameramenu;
%     light
%     camlight head
%     lighting phong	
    
end


