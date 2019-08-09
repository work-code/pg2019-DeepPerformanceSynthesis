function Compute_SuperResolution_Data(p,low_resolution,super_resolution)
p = computer_SuperResolution_SceneName(p);

temp = p.SUPERDATA_SIR;
p = set_root(p,'train'); p.folder_name = p.folder_train_super;
computer_SuperResolution_DataImage(p,low_resolution,super_resolution);
p.SUPERDATA_SIR = temp;

temp = p.SUPERDATA_SIR;
p = set_root(p,'test'); p.folder_name = p.folder_test_super;
computer_SuperResolution_DataImage(p,low_resolution,super_resolution);
p.SUPERDATA_SIR = temp;

end

function p = set_root(p,type)
mkdir(p.SUPERDATA_SIR,type);
p.SUPERDATA_SIR = fullfile(p.SUPERDATA_SIR,type);

mkdir(p.SUPERDATA_SIR,'Scene');
p.SUPERDATA_SCENE_SIR = fullfile(p.SUPERDATA_SIR,'Scene');

mkdir(p.SUPERDATA_SIR,'SuperScene');
p.SUPERDATA_SUPERSCENE_SIR = fullfile(p.SUPERDATA_SIR,'SuperScene');
end

function p = computer_SuperResolution_SceneName(p)
p.folder_train_super = p.folder_train;
p.folder_test_super  = p.folder_test;

end