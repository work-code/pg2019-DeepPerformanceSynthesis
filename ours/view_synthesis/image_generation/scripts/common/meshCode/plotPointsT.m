function h = plotPointsT(vertices,color)
    labels = cellstr(num2str((1:size(vertices,1))')); 
    if nargin==2

    else
        num_vertices = size(vertices,1);
        color = repmat([187/255 1 1],num_vertices,1);
    end
    vv = color;
    %color = (vv-min(vv))./(max(vv)-min(vv)+eps);
    
    h = figure;   
%    scatter3(vertices(:,1),vertices(:,2),vertices(:,3),S,color,'filled');
%    patch('Vertices',vertices,'Faces',faces,'FaceVertexCData',color,'FaceColor','flat');
    shading interp;%shading ;interp FACETED
    view(3); 
	grid on;%view(-6,16); 
    set(gcf,'Renderer','OpenGL');
    alpha(1); axis on; axis image; axis tight; zoom(1); axis off
    lighting flat; %phong; gouraud; flat;
    light('Position',[2 -2 2],'Style','local'); material shiny; %dull, metal infinite shiny
    for k = 1:size(vertices,1)
        plot3(vertices(k,1),vertices(k,2),vertices(k,3),'.','color',color(k,:),'MarkerSize',7);
        hold on
    end
    text(vertices(:,1),vertices(:,2),vertices(:,3),labels);
    axis image off;
    shading flat;
    light;
    camlight head;
    lighting phong;
    material([.5 .5 0.2 25 0]); cameramenu;% colorbar;
    colormap jet;
    	
end