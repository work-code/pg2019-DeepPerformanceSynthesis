function p = setting_up(dataSet,name)

addpath('G:\2-paper\ResearchWork5\ResearchWork5_zz_finalversion\code\ours\view_synthesis\image_generation\scripts\common');
p = setting_upc(dataSet);
p.dataSet = dataSet;

p.folder = {};
count = 1;
for i = 1:size(name,1)
    sub_name = name{i,1};
    for t = 1:size(p.folder_index,1)
        folder_name = strcat(sub_name,'_',int2str(t));
        p.folder{count,1} = folder_name;
        count = count + 1;
    end
end
temp = {};
count = 1;
for i = 1:size(p.folder,1)
    for g = 1:p.frame_size
        i_start = (i-1)*p.frame_size*p.view_size_query + (g-1)*p.view_size_query + 1;
        i_end = (i-1)*p.frame_size*p.view_size_query + g*p.view_size_query;
        temp_name = strcat('Scene_',int2str(i_start),'_',int2str(i_end));
        temp{count,1} = temp_name;
        count = count + 1;
    end
end

p.folder_train = {};
p.folder_test = {};

train_index = compute_index(p.folder_train_index,name,p);
p.folder_train = temp(train_index,1);

test_index = compute_index(p.folder_test_index,name,p);
p.folder_test = temp(test_index,1);
p.folder_val = p.folder_test;
end

function file_index = compute_index(index,name,p)
file_index = [];
for t = 1:size(name,1)
    bias = (t-1)*3*p.frame_size;
    for i = 1:size(index,1)
        temp = index(i,1);
        i_start = bias + (temp-1)*p.frame_size +1 ;
        i_end = bias + temp*p.frame_size;
        temp_index = (i_start:1:i_end)';
        file_index = [file_index;temp_index];
    end
end
end
