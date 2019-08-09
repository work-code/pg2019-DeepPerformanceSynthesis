function saveOFFRGB(fileName,shape)
f = fopen(fileName,'wt');
fprintf(f,'OFF\n');
fprintf(f,'%d %d %d\n',shape.nv,shape.nt,shape.ne);
fprintf(f,'%g %g %g %d %d %d\n',[shape.X shape.Y shape.Z shape.RGB]);
temp = repmat(3,shape.nt,1);
TRIV = [temp,shape.TRIV-1];
fprintf(f,'%d %d %d %d\n',TRIV');
fclose(f);
end