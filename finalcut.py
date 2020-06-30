import matplotlib.pyplot as plt
import random
import bpy
import pandas as pd
import numpy as np
import mathutils

path = "/home/artur/Documents/coords/"
coords=pd.read_csv(path+'points.csv')


coordX = []
coordY = []
coordZ = []
#coordX.append(coords.x.values)
#coordY.append(coords.y.values)
#coordZ.append(coords.z.values)
#plt.grid()
for x, y, z in coords.values:
    print(type(x))
    new_z = z + 1
    #bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, enter_editmode=False, location=(x, y, z))
    coordX.append(x)
    coordY.append(y)
    coordZ.append(new_z)
    

def initialize_metarig():
    bpy.ops.object.armature_basic_human_metarig_add()
    bpy.context.object.data.show_names = True
    skeleton = bpy.data.objects["metarig"]
    vec = mathutils.Vector((0.0, 0.0, -0.9))
    inv = skeleton.matrix_world.copy()
    inv.invert()
# vec aligned to local axis
    vec_rot = vec @ inv
    skeleton.location = skeleton.location + vec_rot


initialize_metarig()


   
bpy.data.objects["metarig"].select_set(True)

bpy.ops.object.mode_set(mode='EDIT')
bpy.context.object.data.use_mirror_x = True
bpy.ops.armature.select_all(action='DESELECT')

ob = bpy.data.objects["metarig"]
arm = ob.data

metarig = [
"spine.006","spine.005","shoulder.R","upper_arm.R","forearm.R","shoulder.L","upper_arm.L","forearm.L","pelvis.R","thigh.R",
"shin.R","pelvis.L","thigh.L","shin.L"
]

def poseHead(metarig_name,blender_pose):
    metarig_pose = arm.edit_bones[metarig_name]
    metarig_pose.head.x = coordX[blender_pose]
    metarig_pose.head.y = coordY[blender_pose]
    metarig_pose.head.z = coordZ[blender_pose]

def poseTail(metarig_name,blender_pose):
    metarig_pose = arm.edit_bones[metarig_name]

    if metarig_name == "foot.L":
        metarig_pose.tail.x = coordX[blender_pose]
        metarig_pose.tail.y = coordY[blender_pose] + 0.3
        metarig_pose.tail.z = coordZ[blender_pose] - 0.2
    elif metarig_name == "toe.L":
        metarig_pose.tail.x = coordX[blender_pose]
        metarig_pose.tail.y = coordY[blender_pose]+0.1
        metarig_pose.tail.z = coordZ[blender_pose]-0.1
    elif metarig_name == "hand.L":
        metarig_pose.tail.x = coordX[blender_pose]
        metarig_pose.tail.y = coordY[blender_pose]+0.1
        metarig_pose.tail.z = coordZ[blender_pose]-0.1
    else:
        metarig_pose.tail.x = coordX[blender_pose]
        metarig_pose.tail.y = coordY[blender_pose]
        metarig_pose.tail.z = coordZ[blender_pose]
        
poseTail("spine.006",0)
poseTail("spine.001",1)
#left side calc.-s
poseHead("upper_arm.L",5)
poseTail("shoulder.L",5)
poseHead("shoulder.L",1)
poseTail("upper_arm.L",6)
poseTail("forearm.L",7)
poseTail("hand.L",7)
poseTail("thigh.L",12)
poseTail("shin.L",13)
poseTail("foot.L",13)
poseTail("toe.L",13)

bpy.ops.object.mode_set(mode='OBJECT')



