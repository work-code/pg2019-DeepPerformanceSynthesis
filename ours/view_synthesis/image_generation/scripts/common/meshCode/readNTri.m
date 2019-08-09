function [vertex,face] = readNTri(filename)

fid = fopen(filename,'r');

str = fgets(fid);

[a,str] = strtok(str); nvert = str2num(a);
[a,str] = strtok(str); nface = str2num(a);



[A,cnt] = fscanf(fid,'%f %f %f %f %f %f', 6*nvert);
if cnt~=6*nvert
    warning('Problem in reading vertices.');
end
A = reshape(A, 6, cnt/6);
vertex = A;
% read Face 1  1088 480 1022
[A,cnt] = fscanf(fid,'%d %d %d %d\n', 4*nface);
if cnt~=4*nface
    warning('Problem in reading faces.');
end
A = reshape(A, 4, cnt/4);
face = A(2:4,:)+1;


fclose(fid);

vertex = vertex';
face = face';
end

