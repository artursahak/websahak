import matplotlib.pyplot as plt
import random
import bpy
import pandas as pd
import numpy as np
import mathutils
import math
import os
import sys
import fnmatch

for eachArg in sys.argv:   
        print(eachArg)

#print(sys.path)

path = "/home/artur/Documents/coords/"
#sys.path.append(path)

print(sys.path)
#2015-01-raw
for c in bpy.data.collections:
    for o in c.objects:
        bpy.data.objects.remove(o)

global unity 
unity = 0.015
def import_obj():
    model_name = "2015-01-raw.obj"
    file_loc = os.path.join(path, model_name)
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    obj_object = bpy.context.selected_objects[0] ####<--Fix
    obj_object.scale = (unity,unity,unity)
    #obj_object.rotation_euler[2]=math.pi
    for obj in bpy.context.selected_objects:
        obj.name = "Model"
        global global_model
        global_model = obj.name
    bpy.ops.object.select_all(action='DESELECT')

import_obj()

#print('Imported name: ', obj_object.name)
excel_name = "points"

coords=pd.read_csv(os.path.join(path, excel_name+".csv"))

coordX = []
coordY = []
coordZ = []

for x, y, z in coords.values:
    print(type(x))
    new_z = z + 0.8
    #bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, enter_editmode=False, location=(x, y, z))
    coordX.append(x)
    coordY.append(-y)
    coordZ.append(new_z)
    #bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05,enter_editmode=False,location=(x,y,new_z))

def initialize_metarig():
    bpy.ops.object.armature_basic_human_metarig_add()
    bpy.context.object.data.show_names = True
    skeleton = bpy.data.objects["metarig"]
    vec = mathutils.Vector((-0.033, 0.0, 0.52))
    inv = skeleton.matrix_world.copy()
    inv.invert()
    print(inv)

# vec aligned to local axis
    vec_rot = vec @ inv
    
    skeleton.location = skeleton.location + vec_rot

initialize_metarig()

bpy.data.objects["metarig"].select_set(True)

bpy.ops.object.mode_set(mode='EDIT')
#bpy.context.object.data.use_mirror_x = True
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
    if metarig_name == "spine.006":
        metarig_pose.tail.x = coordX[blender_pose]
        metarig_pose.tail.y = coordY[blender_pose]+0.2
        metarig_pose.tail.z = coordZ[blender_pose]
    elif metarig_name == "foot.L" or metarig_name == "foot.R":
        metarig_pose.tail.x = coordX[blender_pose]
        metarig_pose.tail.y = coordY[blender_pose] -0.3
        metarig_pose.tail.z = coordZ[blender_pose] 
    elif metarig_name == "toe.L" or metarig_name == "toe.R":
        metarig_pose.tail.x = coordX[blender_pose]
        metarig_pose.tail.y = coordY[blender_pose]-0.1
        metarig_pose.tail.z = coordZ[blender_pose]
    elif metarig_name == "forearm.L" or metarig_name == "forearm.R":
        metarig_pose.tail.x = coordX[blender_pose]
        metarig_pose.tail.y = coordY[blender_pose]-0.02
        metarig_pose.tail.z = coordZ[blender_pose]          
    elif metarig_name == "hand.L" or metarig_name == "hand.R":
        metarig_pose.tail.x = coordX[blender_pose]
        metarig_pose.tail.y = coordY[blender_pose]-0.02
        metarig_pose.tail.z = coordZ[blender_pose]-0.15
    else:
        metarig_pose.tail.x = coordX[blender_pose]
        metarig_pose.tail.y = coordY[blender_pose]
        metarig_pose.tail.z = coordZ[blender_pose]
        
for bone in arm.edit_bones:
    if fnmatch.fnmatchcase(bone.name,"heel.02.L") or fnmatch.fnmatchcase(bone.name,"heel.02.R"): 
        arm.edit_bones.remove(bone)
      
poseTail("spine.006",0)
#poseTail("spine.001",1)
#left side calc.-s
poseHead("upper_arm.L",5)
poseTail("upper_arm.L",6)
poseTail("shoulder.L",5)
poseHead("shoulder.L",1)
poseTail("forearm.L",7)
poseTail("hand.L",7)
poseTail("thigh.L",12)
poseTail("shin.L",13)
poseTail("foot.L",13)
poseTail("toe.L",13)

#right side calc.-s

poseHead("upper_arm.R",2)
poseTail("upper_arm.R",3)
poseTail("shoulder.R",2)
poseHead("shoulder.R",1)
poseTail("forearm.R",4)
poseTail("hand.R",4)
poseTail("thigh.R",9)
poseTail("shin.R",10)
poseTail("foot.R",10)
poseTail("toe.R",10)



bpy.ops.object.mode_set(mode='OBJECT')

objects = bpy.data.objects
objects['metarig'].select_set(True)
objects[global_model].select_set(True)
        #a.parent = b
bpy.ops.object.parent_set(type='ARMATURE_AUTO')

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(path, "project.blend"))














