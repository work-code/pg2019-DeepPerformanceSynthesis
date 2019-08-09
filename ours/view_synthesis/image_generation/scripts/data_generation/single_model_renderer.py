import sys
import bpy
import math
#import scipy.io as sio


argv = sys.argv
argv = argv[argv.index("--") + 1:]

camera_observation = int(argv[0])
camera_query = int(argv[1])
camera_bias = float(argv[2])
resolution_x = int(argv[3])
resolution_y = int(argv[4])
model_location = argv[5]
output_dir = argv[6]
camerac_dir = argv[7]

# load the empty scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Number of Observation View
NV = camera_observation

# Number of Query View
NQ = camera_query

# bias degree
bias = camera_bias

# iput model
bpy.ops.import_scene.obj(filepath = model_location)

# camera calibration
camera_location = []
camera_rotation = []

# set Observation Cameras
for n in range(0, NV):
    tlocation = ((1.4*math.cos(n/NV*math.pi*2+bias))/1.35, (1.4*math.sin(n/NV*math.pi*2+bias))/1.35, 0.375)
    trotation = (1.5*math.pi, math.pi, 1.5*math.pi + n/NV*math.pi*2 + bias)
    bpy.ops.object.camera_add(location = tlocation, 
                              rotation = trotation, 
                              layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
    camera_location.append(tlocation)
    camera_rotation.append(trotation)
    
# set Query Cameras
for n in range(0, NQ):
    tlocation = ((1.4*math.cos(n/NQ*math.pi*2+bias))/1.35, (1.4*math.sin(n/NQ*math.pi*2+bias))/1.35, 0.375)
    trotation = (1.5*math.pi, math.pi, 1.5*math.pi + n/NQ*math.pi*2 + bias)
    bpy.ops.object.camera_add(location = tlocation, 
                              rotation = trotation, 
                              layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
    camera_location.append(tlocation)
    camera_rotation.append(trotation)
	
# set light
bpy.ops.object.lamp_add(type='HEMI', radius=1, view_align=False, location=(0.0, 0.0, 5.0))

# render image, (four channel: R/G/B/Trans)
for m in range(0, NV+NQ):
    if m==0:
        bpy.context.scene.camera = bpy.data.objects['Camera']
    else:
        bpy.context.scene.camera = bpy.data.objects['Camera.%03d' %m ]
    bpy.context.scene.render.resolution_x = resolution_x*2
    bpy.context.scene.render.resolution_y = resolution_y*2
    bpy.data.scenes["Scene"].render.alpha_mode = 'TRANSPARENT'
    bpy.data.scenes["Scene"].render.filepath = '{}{}.jpg'.format(output_dir, m+1)
    bpy.ops.render.render(write_still = True, use_viewport = True, layer=("0"))
    

# register in a file when one batch is ran through
file_name = camerac_dir + 'camera_location.txt'   
f = open(file_name,'w')
for i in range(len(camera_location)):
    temp = camera_location[i]
    for g in range(len(temp)):
        f.write(str(temp[g]))
        if g <= len(temp):
            f.write(' ')
    f.write('\n')
f.close()

file_name = camerac_dir + 'camera_rotation.txt'   
f = open(file_name,'w')
for i in range(len(camera_rotation)):
    temp = camera_rotation[i]
    for g in range(len(temp)):
        f.write(str(temp[g]))
        if g <= len(temp):
            f.write(' ')
    f.write('\n')
f.close()