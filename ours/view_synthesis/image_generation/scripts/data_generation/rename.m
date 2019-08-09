clear  
clc  
  
cd('E:\3_researchWork5\VideoMorphing\TDataGeneration\data\PVHM\rendering\test\cameracalibration\')
parfor i = 1:2000
   oldname = strcat('Scene',int2str(8000 + i));
   newname = strcat('Scene',int2str(2000 + i));
   
   movefile(oldname,newname); 
end

% cd('E:\3_researchWork5\VideoMorphing\TDataGeneration\data\PVHM\rendering\test\scene\')
% dirOutput = dir(fullfile('E:\3_researchWork5\VideoMorphing\TDataGeneration\data\PVHM\rendering\test\scene\'));
% file_name ={dirOutput.name}';
% for i = 3:size(file_name,1)
%     file = char(file_name{i,1});
%     index = str2num(file(1,6:size(file,2)));
%     if index > 9000
%         oldname = file;
%         newname = strcat('Scene',int2str(index -6000));
%         movefile(oldname,newname); 
%     end
%     
% end
