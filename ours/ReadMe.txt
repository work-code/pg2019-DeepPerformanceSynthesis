Hi there, thank you for your attention of our work.

This folder contains the source codes used in our paper.

I will give a detailed introduction of the folder's directory structure and environment condiguration of these codes.

*******************************************************************
Folder's directory structure

                       Folder Name                                               Function

1. view_synthesis --- image_generation --- scripts --- data_generation            Rendering images for *NOVEL-VIEW PREDICTION* and *DETAIL ENHANCEMENT*          
                   
                      image_generation --- scripts --- data_convert				  Converting images's (used in *NOVEL-VIEW PREDICTION*) format: from '.jpg' to '.pt'

2. view_synthesis --- view_synthesis   --- scripts                                Performing the training and testing processes of *NOVEL-VIEW PREDICTION*

                      view_synthesis   --- scripts2                               Computing the dataset used for 'fine-tuning' process 
 
3. super_resolution --- DataVonvert    --- scripts                                Converting images's (used in *DETAIL ENHANCEMENT*) format: from '.jpg' to '.pt'

4. super_resolution --- SRResNet       --- scripts                                Performing the training and testing processes of *DETAIL ENHANCEMENT*

                    --- SRResNet       --- scripts_retrain                        Performing the fine-tuning process of *DETAIL ENHANCEMENT*
*******************************************************************

*******************************************************************
Environment condiguration

1. Python 3.6.3

2. PyTorch 0.4.1

3. blender-2.79b

4. Matlab 2017a
*******************************************************************

In these codes, the path for reading the data is temporarily set as our local path.

Please remember to change it to your local path.

We have made detailed comments in the code.

If you have any questions, please don't hesitate to contact us.

You can open an issue through the button "Issues" on the github.

Thank you for your kind attention again.