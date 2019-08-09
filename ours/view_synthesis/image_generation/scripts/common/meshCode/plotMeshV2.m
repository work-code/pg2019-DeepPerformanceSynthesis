function h = plotMeshV2(vertices,faces,pVertices,color)

    if nargin==4

    else
        num_vertices = size(vertices,1);
%        color = ones(num_vertices,1);
        color = repmat([187/255 1 1],num_vertices,1);
    end
    vv = color;
    %color = (vv-min(vv))./(max(vv)-min(vv)+eps);
    h = figure('color','w','renderer','zbuffer','units','pixels','position',[100 100 600 600]);
%    h = figure;   
    hold on;
    
    
    patch('Vertices',vertices,'Faces',faces,'FaceVertexCData',color,'FaceColor','flat');
    shading INTERP;%FACETED;INTERP
    view(3); grid on; %view(-6,16); 
    set(gcf,'Renderer','OpenGL');
    alpha(1); axis on; axis image; axis tight; zoom(1); axis off
    lighting phong; %phong; gouraud; flat;
    light('Position',[-2 -2 2],'Style','infinite'); material shiny; %dull, metal shiny infinite
    material([.5 .5 0.2 25 0]); cameramenu; %colorbar; 
%    colormap jet;
%     x = -2:2;
%     y = 0;
%     z = 0;,'c.','MarkerSize',15
%     line([0 1],[0 0],[0 0],'linewidth',4,'color','r');
%     line([0 0],[0 1],[0 0],'linewidth',4,'color','r');
%     line([0 0],[0 0],[0 1],'linewidth',4,'color','r');

    if size(pVertices,1) ~= 0
       plot3(pVertices(:,1),pVertices(:,1),pVertices(:,1),'b.','MarkerSize',35); % visualize the point cloud
    end
%    plot3(vertices(606,1),vertices(606,2),vertices(606,3),'g.','MarkerSize',55);
%    plot3(vertices(5360,1),vertices(5360,2),vertices(5360,3),'r.','MarkerSize',55);
    hold on; % so we can highlight the clicked points without clearing the point cloud
    cameramenu;
    %plot3(vertices(:,1),vertices(:,2),vertices(:,3),'r','MarkerSize',0.5);
    
end


