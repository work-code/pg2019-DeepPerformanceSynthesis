function h = plotSphere(color2)

h = figure('color','w','renderer','zbuffer','units','pixels','position',[100 100 600 600]);
sphere = TriangulatedSphere(50);

axisPoint1 = [0,0,0];
sphere1 = sphere;
sphere1.X = 4*sphere.X + axisPoint1(1,1);
sphere1.Y = 4*sphere.Y + axisPoint1(1,2);
sphere1.Z = 4*sphere.Z + axisPoint1(1,3);
trisurf(sphere1.TRI,sphere1.X,sphere1.Y,sphere1.Z,'facecolor',color2,'edgecolor','none');
hold on;
set(gcf,'Renderer','OpenGL');
    alpha(1); axis on; axis image; axis tight; zoom(1); axis off
    lighting phong; %phong; gouraud; flat;
    light('Position',[-2 2 2],'Style','local'); material shiny;
    material([.5 .5 0.2 25 0]); cameramenu;
end