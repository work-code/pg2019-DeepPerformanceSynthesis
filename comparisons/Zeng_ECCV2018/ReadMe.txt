Hi there, thank you for your kind attention of our project.

This folder contains the comparison code [Zeng et al. ECCV2018] used in our paper.

[Zeng et al. ECCV2018]: Huang Z, Li T, Chen W, et al. Deep volumetric video from very sparse multi-view performance capture. ECCV. 2018.

We will give a detailed introduction of the folder's directory structure and environment condiguration of this code.

This code is provided by the original author of paper [Zeng et al. ECCV2018].

*******************************************************************
Folder's directory structure
            
   File Name                                                 Function
			
1. color.py                                                  Used to colorize each vertex of the model

2. delete_point.py                                           Used to process the point cloud away from the model

3. isinsider.py                                              Used to distinguish the location of the point (inside of the model, locate on the surface of the model or outside of the model)

4. length.py                                                 Used to compute the distence between the point to the model

5. marching_cube.py                                          Used to reconstruct the 3D model

6. Net.py                                                    Used to define the network

7. projection.py                                             Used to project the point onto the images

8. qr.py                                                     Used to process the matrix of the camera parameter

9. txt_obj.py & obj_ply.m                                    Used to transform the model's format

10. image_renderer.m  & single_model_renderer.py             Used for the rendering of the model

11. render.m                                                 Used to generate the final rendered model

12. camera_site.txt                                          Used to save the camera parameter

13. train.py                                                 Used for the training process
*******************************************************************

*******************************************************************
Environment condiguration

1. Python 3.6.3

2. conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

3. blender-2.79b

4. Matlab 2017a

5. Meshlab
*******************************************************************

*******************************************************************
Code usage steps

1. Download the dataset from the link: https://drive.google.com/drive/folders/1bvY6IMSn1whv6C0fcAaxglkfm77My-Wt?usp=sharing

2. Using image_renderer.m to render the images

3. Using train.py to train the model

4. Using marching_cube.py to reconstruct the 3D model, that is the testing process

5. Using color.py to colorize each vertex of the model

6. Using txt_obj.py & obj_ply.m  to transform the model's format

7. Using renderer.m to to generate the final rendered model
*******************************************************************

There are some things should be awared:

1. There are some codes needed to be run in segments, and we have made comment to label these codes.

2. In this code, the path for reading the data is temporarily set as our local path.

   Please remember to change it to your local path.
   
   We have made comments in the code.
   
3. There are also some related folders should be built manually.
   
   And these folders have been noted in the code.
   
If you have any questions, please don't hesitate to contact us.

You can open an issue through the button "Issues" on the github.

Thank you for your kind attention again.