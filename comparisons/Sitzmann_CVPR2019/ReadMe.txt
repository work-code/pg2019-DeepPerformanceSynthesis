Hi there, thank you for your kind attention of our project.

This folder contains the comparison code [Sitzmann et al. CVPR2019] used in our paper.

[Sitzmann et al. ECCV2018]: Sitzmann, Vincent, et al. Deepvoxels: Learning persistent 3d feature embeddings. CVPR. 2019.

We will give a detailed introduction of the folder's directory structure and environment condiguration of this code.

This code is provided by the original author of paper [Sitzmann et al. CVPR2019], the link is: https://github.com/vsitzmann/deepvoxels.

*******************************************************************
Folder's directory structure
            
   File Name                                                 Function
			
1. dataio.py                                                 Used to load the training and testing data

2. data_util.py & util.py                                    Used to contain the utility functions

3. deep_voxels.py                                            Used to define the DeepVoxels model

4. custom_layers.py                                          Used to define the integration and occlusion submodules

5. projection.py                                             Used to contain utility functions for 3D and projective geometry

6. run_deepvoxels.py                                         Used to train and test the network

*******************************************************************

*******************************************************************
Environment condiguration

1. Python 3.7

2. PyTorch 1.0

3. conda env create -f environment.yml

*******************************************************************

*******************************************************************
Code usage steps

1. Download the dataset from the link: https://drive.google.com/drive/folders/1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH

2. Training: 

              See python run_deepvoxels.py --help for all train options. Example train call:
			  
			  python run_deepvoxels.py --train_test train \
                         --data_root [path to directory with dataset] \
                         --logging_root [path to directory where tensorboard summaries and checkpoints should be written to] 
			
              To monitor progress, the training code writes tensorboard summaries every 100 steps into a "runs" subdirectory in the logging_root.
			  
3. Testing:   

              Example test call:
			  
			  python run_deepvoxels.py --train_test test \
                         --data_root [path to directory with dataset] ]
                         --logging_root [path to directoy where test output should be written to] \
                         --checkpoint [path to checkpoint]

*******************************************************************

Also, there has a "README.md" file in the directory "\Sitzmann_CVPR2019\scripts", 

this file is provided by the original author.
   
If you have any questions, please don't hesitate to contact us.

You can open an issue through the button "Issues" on the github.

Thank you for your kind attention again.