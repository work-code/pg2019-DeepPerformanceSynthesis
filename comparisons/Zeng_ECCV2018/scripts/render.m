
set(0,'defaultfigurecolor','black');

%resolution=512
for model=9296:9399
    width=500/781*512;
    height=500/781*512;
    bias=pi/36 * -36
    NV=10
    model_path=sprintf('C:\\Users\\Desktop\\Project9-contrast\\Project2\\color_points_ply\\%d.ply',model)
    ptCloud=pcread(model_path)
    for n=0:2*NV-1 
     
        h=figure('Visible','off')

        pcshow(ptCloud)
        campos([1.4*cos(n/NV*pi+bias), 1.4*sin(n/NV*pi+bias),0.4])
        axis off
        camva(35.5)
        camtarget([0,0,0.4])
        set(gcf, 'InvertHardCopy', 'off');
        set(gcf,'position',[100,100,width ,height ])

        path=sprintf('C:\\Users\\Desktop\\Project9-contrast\\image_renderer\\output_image_512\\%d_%d.png',model,n)
        saveas(gcf,path)
    end

   
    close all

end 



