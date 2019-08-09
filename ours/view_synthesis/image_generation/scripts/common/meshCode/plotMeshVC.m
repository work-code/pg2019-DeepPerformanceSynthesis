function h = plotMeshVC(vertices,faces,pVertices,color2,color)

    if nargin==5

    else
        num_vertices = size(vertices,1);
%        color = ones(num_vertices,1);
        %color = repmat([187/255 1 1],num_vertices,1);
        color = repmat([245/255 245/255 245/255],num_vertices,1);
    end
    vv = color;
    %color = (vv-min(vv))./(max(vv)-min(vv)+eps);
    h = figure('color','w','renderer','zbuffer','units','pixels','position',[100 100 600 600]);

    
    
    patch('Vertices',vertices,'Faces',faces,'FaceVertexCData',color,'FaceColor','flat');
    axis image off;
    shading flat
    hold on;
    for i = 1:size(pVertices,1)
        axisPoint1 = vertices(pVertices(i,:),:);
        shape = loadFile('E:\2_researchWork4\2_CMC_ACC\3_Airplane\LocalPatch\data\Airplane\figure\cube.mat');

        shape.X = 0.01*shape.X + axisPoint1(1,1);
        shape.Y = 0.01*shape.Y + axisPoint1(1,2);
        shape.Z = 0.01*shape.Z + axisPoint1(1,3);
        trisurf(shape.TRIV,shape.X,shape.Y,shape.Z,'facecolor',color2(i,:),'edgecolor','none');
        hold on;
    end 
    set(gcf,'Renderer','OpenGL');
    alpha(1); axis on; axis image; axis tight; zoom(1); axis off
    lighting phong; %phong; gouraud; flat;
    light('Position',[-2 2 2],'Style','local'); material shiny;
    material([.5 .5 0.2 25 0]); cameramenu;
%     light
%     camlight head
%     lighting phong	
    
end


