function savePly(fileName,shape)
f = fopen(fileName,'wt');
fprintf(f,'ply\n');
fprintf(f,'format ascii 1.0\n');
fprintf(f,'element vertex %d\n',shape.nv);
fprintf(f,'property float x\n');
fprintf(f,'property float y\n');
fprintf(f,'property float z\n');
fprintf(f,'element face %d\n',shape.nt);
fprintf(f,'property list uchar int vertex_indices\n');
fprintf(f,'end_header\n');

fprintf(f,'%g %g %g\n',[shape.X shape.Y shape.Z]');
temp = repmat(3,shape.nt,1);
TRIV = [temp,shape.TRIV-1];
fprintf(f,'%d %d %d %d\n',TRIV');
fclose(f);
end