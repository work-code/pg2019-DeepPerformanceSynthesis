function saveOff(fileName,shape)

f = fopen(fileName,'wt');
fprintf(f,'OFF\n');
shape.nv = size(shape.X,1);
shape.nt = size(shape.TRIV,1);
shape.ne = size(compute_edges(shape.TRIV'),1);
fprintf(f,'%d %d %d\n',shape.nv,shape.nt,shape.ne);
fprintf(f,'%g %g %g\n',[shape.X shape.Y shape.Z]');
temp = repmat(3,shape.nt,1);
TRIV = [temp,shape.TRIV-1];
fprintf(f,'%d %d %d %d\n',TRIV');
fclose(f);
end