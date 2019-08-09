function h = plotCube(color2)

h = figure('color','w','renderer','zbuffer','units','pixels','position',[100 100 600 600]);
        shape = loadFile('E:\2_researchWork4\2_CMC_ACC\3_Airplane\LocalPatch\data\Airplane\figure\cube.mat');

        shape.X = 0.1*shape.X; 
        shape.Y = 0.1*shape.Y;
        shape.Z = 0.1*shape.Z;
        trisurf(shape.TRIV,shape.X,shape.Y,shape.Z,'facecolor',color2,'edgecolor','none');
        hold on;
        
set(gcf,'Renderer','OpenGL');
    alpha(1); axis on; axis image; axis tight; zoom(1); axis off
    lighting phong; %phong; gouraud; flat;
    light('Position',[-2 2 2],'Style','local'); material shiny;
    material([.5 .5 0.2 25 0]); cameramenu;
end