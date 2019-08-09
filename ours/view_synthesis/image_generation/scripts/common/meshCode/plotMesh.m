function h = plotMesh(vertices,faces,camera_position)


    num_vertices = size(vertices,1);
    color = repmat([187/255 1 1],num_vertices,1);

    %color = (vv-min(vv))./(max(vv)-min(vv)+eps);
    
    h = figure;   
    patch('Vertices',vertices,'Faces',faces,'FaceVertexCData',color,'FaceColor','flat');
    shading interp;%shading ;interp FACETED
    view(3); 
	grid on;%view(-6,16); 
    set(gcf,'Renderer','OpenGL','position',[100,100,640,640]);
    alpha(1); axis on; axis image; axis tight; zoom(1); axis off
    lighting gouraud; %phong; gouraud; flat;
    light('Position',[4 4 4],'Style','infinite'); material metal; %dull, metal infinite
    %light('Position',[-1 -1 1],'Style','local'); material shiny; %dull, metal infinite
    material([.5 .5 0.2 25 0]); cameramenu;% colorbar;
    view(camera_position);
    %view
    colormap jet;
    	
end