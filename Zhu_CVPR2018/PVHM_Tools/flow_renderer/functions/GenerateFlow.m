
MeshFolderPath='D:\MatlabCode\Flow\one_tuple_for_pose_flow_generate_test\';
OutputMeshPath='D:\MatlabCode\Flow\one_tuple_for_pose_flow_generate_test\output\';

Tpose_name='Tpose.obj';
Pose1_name='pose_1.obj';

Tpose_name_ply='Tpose.ply';
Pose1_name_ply='pose_1.ply';

%% view 
K =[ 245.0000,   0.0000, 112.0000;
   0.0000, 245, 112.0000;
    0.0000,   0.0000,   1.0000];

Rt0=[-0.3420,  0.9397,  0.0000, -0.00 ;
-0.0000, -0.0000, -1.0000,  0.4000; 
 -0.9397, -0.3420, -0.0000,  1.4000];



[flow_x,flow_y]=poseflow(MeshFolderPath,OutputMeshPath,Tpose_name,Pose1_name,Tpose_name_ply,Pose1_name_ply,K,Rt0);






