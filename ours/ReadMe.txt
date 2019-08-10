Hi there, thank you for your attention of our work.

This folder contains the source codes used in our paper.

We will give a detailed introduction of the folder's directory structure and environment condiguration of these codes.

*******************************************************************
Folder's directory structure

                       Folder Name                                               Function

1. view_synthesis --- image_generation --- scripts --- data_generation      Rendering images for *NOVEL-VIEW PREDICTION* and *DETAIL ENHANCEMENT*

                      image_generation --- scripts --- data_convert        Converting images's (used in *NOVEL-VIEW PREDICTION*) format: from '.jpg' to '.pt'

2. view_synthesis --- view_synthesis   --- scripts                          Performing the training and testing processes of *NOVEL-VIEW PREDICTION*

                      view_synthesis   --- scripts2                         Computing the dataset used for 'fine-tuning' process 
 
3. super_resolution --- DataVonvert    --- scripts                          Converting images's (used in *DETAIL ENHANCEMENT*) format: from '.jpg' to '.pt'

4. super_resolution --- SRResNet       --- scripts                          Performing the training and testing processes of *DETAIL ENHANCEMENT*

                    --- SRResNet       --- scripts_retrain                  Performing the fine-tuning process of *DETAIL ENHANCEMENT*
*******************************************************************

*******************************************************************
Environment condiguration

1. Python 3.6.3

2. PyTorch 0.4.1

3. blender-2.79b

4. Matlab 2017a
*******************************************************************

*******************************************************************
Code usage steps:

#################################################### Data Processing #########################################################

1. Download the dataset from the link: https://drive.google.com/drive/folders/1bvY6IMSn1whv6C0fcAaxglkfm77My-Wt?usp=sharing

2. Using \view_synthesis\image_generation\scripts\data_generation to rendering images used for training and testing

3. Using \view_synthesis\image_generation\scripts\data_convert to convert image's format from '.jpg' to '.pt'

4. Moving the folder "torch" from "\view_synthesis\image_generation\data\PVHM" to "\view_synthesis\view_synthesis\data\PVHM"



################################ Train the first network (*NOVEL-VIEW PREDICTION*) ###########################################

5. Using \view_synthesis\view_synthesis\scripts to train the first network

6. Using \view_synthesis\view_synthesis\scripts2 to genrate the data used for fine-tuning



################################ Train the second network (*DETAIL ENHANCEMENT*) #############################################

7. Moving the folder "superdata" from "\view_synthesis\image_generation\data\PVHM" to "\super_resolution\DataConvert\data\PVHM"

8. Using "\super_resolution\DataConvert\scripts" to convert image's format from '.jpg' to '.pt'

9. Moving the folder "torch" from "\super_resolution\DataConvert\data\PVHM" to "\super_resolution\SRResNet\data\PVHM"

10.Using "\super_resolution\SRResNet\scripts" to train the second network



################################ Run the fine-tuning process (*DETAIL ENHANCEMENT*) ###########################################

11.Moving "torch_super" from "\view_synthesis\view_synthesis\data\PVHM" to "\super_resolution\SRResNet\data\PVHM"

12.Moving the last generated model in the directory "\super_resolution\SRResNet\data\PVHM\model" to "super_resolution\SRResNet\data\PVHM\retrain\model"

13.Using "\super_resolution\SRResNet\scripts_retrain" to run the fine-tuning process


*******************************************************************



In these codes, the path for reading the data is temporarily set as our local path.

Please remember to change it to your local path.

We have made detailed comments in the code.

If you have any questions, please don't hesitate to contact us.

You can open an issue through the button "Issues" on the github.

Thank you for your kind attention again.
