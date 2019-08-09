function saveRawData(rawData, root, shapeName, fileName)
mkdir(root,shapeName);
imageFile = fullfile(root,shapeName,strcat(fileName,'.mat'));
shape = rawData;
save(imageFile,'shape');
end