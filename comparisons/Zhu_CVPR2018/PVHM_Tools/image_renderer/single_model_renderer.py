import sys
import bpy
import math

argv = sys.argv
argv = argv[argv.index("--") + 1:]

model_index = int(argv[0])
resolution_x = int(argv[1])
resolution_y = int(argv[2])
model_location = argv[3]
output_dir = argv[4]

# load the empty scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Number of View
NV = 18

# bias degree
bias = math.pi/36 * -36

# iput model
bpy.ops.import_scene.obj(filepath = model_location)

# set Cameras
for n in range(0, NV+1):
    bpy.ops.object.camera_add(location = (1.4*math.cos(n/NV*math.pi+bias), 1.4*math.sin(n/NV*math.pi+bias), 0.40), rotation = (1.5*math.pi, math.pi, 1.5*math.pi + n/NV*math.pi + bias), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))

# set light
bpy.ops.object.lamp_add(type='HEMI', radius=1, view_align=False, location=(0.0, 0.0, 5.0))

# render image, (four channel: R/G/B/Trans)
for m in range(0, NV+1):
    if m==0:
        bpy.context.scene.camera = bpy.data.objects['Camera']
    else:
        bpy.context.scene.camera = bpy.data.objects['Camera.%03d' %m ]
    bpy.context.scene.render.resolution_x = resolution_x*2
    bpy.context.scene.render.resolution_y = resolution_y*2
    bpy.data.scenes["Scene"].render.alpha_mode = 'TRANSPARENT'
    bpy.data.scenes["Scene"].render.filepath = '{}{}_{}_{}x{}.jpg'.format(output_dir, model_index, m, resolution_x, resolution_y)
    bpy.ops.render.render(write_still = True, use_viewport = True, layer=("0"))

# register in a file when one batch is ran through
f = open('./rendered_list_tmp.txt','a')
f.write('{}\n'.format(model_index))
f.close()