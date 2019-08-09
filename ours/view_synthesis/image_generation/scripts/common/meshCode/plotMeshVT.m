function plotMeshVT(vertices,faces,pVertices)

    labels = cellstr(num2str(pVertices)); 
    num_vertices = size(vertices,1);
    color = ones(num_vertices,1);

    vv = color;
    %color = (vv-min(vv))./(max(vv)-min(vv)+eps);
    
    figure   
    hold on;
    patch('Vertices',vertices,'Faces',faces,'FaceVertexCData',color,'FaceColor','flat');
    shading interp;
    view(3); grid on; %view(-6,16); 
    set(gcf,'Renderer','OpenGL');
    alpha(1); axis on; axis image; axis tight; zoom(1.2); axis off
    lighting phong; %phong; gouraud; flat;
    light('Position',[-1 -1 1],'Style','infinite'); material shiny; %dull, metal
    material([.5 .5 0.2 25 0]); cameramenu; colorbar;
    

    
    plot3(vertices(pVertices(:,1),1),vertices(pVertices(:,1),2),vertices(pVertices(:,1),3),'c.','MarkerSize',15); % visualize the point cloud
    text(vertices(pVertices(:,1),1),vertices(pVertices(:,1),2),vertices(pVertices(:,1),3),labels);
    hold on; % so we can highlight the clicked points without clearing the point cloud
    cameramenu;
    %plot3(vertices(:,1),vertices(:,2),vertices(:,3),'r','MarkerSize',0.5);
end


