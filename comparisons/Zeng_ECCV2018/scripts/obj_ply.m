for model=9296:9399
    mesh_obj_filename= sprintf('C:\\Users\\Desktop\\Project9-contrast\\Project2\\color_points_obj\\%d.obj',model)
    mesh_ply_filename= sprintf('C:\\Users\\Desktop\\Project9-contrast\\Project2\\color_points_ply\\%d.ply',model)
    [~,~]=system(sprintf('C:\\PROGRA~1\\VCG\\MeshLab\\meshlabserver.exe  -i %s -o %s -m vc ',mesh_obj_filename,mesh_ply_filename));
end
 