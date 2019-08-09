function plotPointsV(vertices,pVertices,color)

    if nargin==3

    else
        num_vertices = size(vertices,1);
        color = repmat([187/255 1 1],num_vertices,1);
    end
    vv = color;
    %color = (vv-min(vv))./(max(vv)-min(vv)+eps);
    
    h = figure;   
%    patch('Vertices',vertices,'Faces',faces,'FaceVertexCData',color,'FaceColor','flat');
    shading FACETED;%shading ;interp FACETED
    view(3); 
	grid on;%view(-6,16); 
    set(gcf,'Renderer','OpenGL');
    alpha(1); axis on; axis image; axis tight; zoom(1); axis off
    lighting flat; %phong; gouraud; flat;
    light('Position',[2 -2 2],'Style','local'); material shiny; %dull, metal infinite
    plot3(vertices(:,1),vertices(:,2),vertices(:,3),'b.','MarkerSize',7); % visualize the point cloud
    hold on;
    plot3(vertices(pVertices,1),vertices(pVertices,2),vertices(pVertices,3),'r.','MarkerSize',7); % visualize the point cloud
    material([.5 .5 0.2 25 0]); cameramenu;% colorbar;
    colormap jet;
end