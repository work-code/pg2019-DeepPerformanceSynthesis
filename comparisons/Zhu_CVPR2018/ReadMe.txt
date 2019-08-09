Hi there, thank you for your kind attention of our project.

This folder contains the comparison code [Zhu et al. CVPR2018] used in our paper.

[Zhu et al. CVPR2018]Zhu, Hao, et al. View extrapolation of human body from a single image. CVPR. 2018.

We will give a detailed introduction of the folder's directory structure and environment condiguration of this code.

This data is provided by the original author of paper [Zhu et al. CVPR2018].

*******************************************************************
Folder's directory structure
            
   Folder Name                                               Function
			
1. PVHM_Tools                                                Rendering the dataset used to run the network

2. VSPV_train                                                Performing the training process
              
			  --- model                                      The customization layer of the network
			  
			  --- snapshot                                   This folder is used to store model weights
			  
			  --- solver                                     This folder contains three network models used in the paper
			  
			  --- utility                                    This folder contains the functions used to running this network
			  
			  --- project_functions.py                       Used for the forward_flow 
			  
			  --- remap_functions.py                         Used to reset the image's resolution
			  
			  --- forward_flow.py                            Used to generate the forward_flow, forward_mask and reampped_forward_flow
			  
			  --- Tuple_build_clean.py                       Used to output the path (or directory) of the training dataset
			  
			  --- **_loss_**.txt                             Used to stoarge the losses generated during the training process

			  --- train.py                                   Used for the training process
			  
3. VSPV_test                                                 Performing the testing process
          
		      --- final_image                                Used to storage the final output results
			  
			  --- resolution.py                              Used to reset the resolution of the final outputs
			  
			  --- my_test.py                                 Used for the testing process
*******************************************************************

*******************************************************************
Environment condiguration

1. Ununtu+caffe+cuda8.0+python2.7

3. blender-2.79b

4. Matlab 2017a
*******************************************************************

*******************************************************************
Code usage steps

1. Download the dataset from the link: https://drive.google.com/drive/folders/1bvY6IMSn1whv6C0fcAaxglkfm77My-Wt?usp=sharing

2. Using "PVHM_Tools" to generate the backward_flow, rendering image and depth image

3. Using "forward_flow.py" to generate the forward_flow, forward_mask and reampped_forward_flow

4. Using "Tuple_build_clean.py" to compute the file (.tex), which is used to stoarge the path of the dataset

5. Training the network using "train.py"

6. Using the testing script "my_test.py" to generate the mask and the backward_flow

7. Uisng the script "resolution.py" and the generated mask and backward_flow to compute the final results
*******************************************************************

There are some things should be awared:

1. In these codes, the path for reading the data is temporarily set as our local path.

   Please remember to change it to your local path.
   
   We have made comments in the code.
   
2. In the testing process, you should move the trained model (that is, the folder "snapshot") from VSPV_train to VSPV_test manually.

3. There are also some related folders should be built manually.
   
   And these folders have been noted in the code.
   
If you have any questions, please don't hesitate to contact us.

You can open an issue through the button "Issues" on the github.

Thank you for your kind attention again.