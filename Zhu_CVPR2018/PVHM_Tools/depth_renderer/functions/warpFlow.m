function [warpI2,I]=warpFlow(i2,vx,vy)
[M,N]=size(i2);
[x,y]=meshgrid(1:N,1:M);
warpI2=interp2(x,y,i2,x+vx,y+vy,'bicubic');
I=find(isnan(warpI2));
warpI2(I)=zeros(size(I));