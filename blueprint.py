
import bpy

#print("dlib-",dlib.__version__,"cv2-",cv2.__version__,"numpy-",np.__version__)
import random
import mathutils

def initialize_metarig():
    bpy.ops.object.armature_basic_human_metarig_add()
    bpy.context.object.data.show_names = True

initialize_metarig()
   
bpy.data.objects["metarig"].select_set(True)

bpy.ops.object.mode_set(mode='EDIT')
bpy.context.object.data.use_mirror_x = True
bpy.ops.armature.select_all(action='DESELECT')

ob = bpy.data.objects["metarig"]
arm = ob.data
#mod = 60

#upperarmX = converter[6][0]
#upperarmZ = converter[6][1]

upper_armL=arm.edit_bones['upper_arm.L']
upper_armL.tail.x = 0.39
upper_armL.tail.y = 0.15
upper_armL.tail.z = 1.4

fore_armL = arm.edit_bones['forearm.L']
fore_armL.tail.x = 0.42
fore_armL.tail.y = -0.015
fore_armL.tail.z = 1.13

handL = arm.edit_bones['hand.L']
handL.tail.x = 0.42
handL.tail.y = -0.015
handL.tail.z = 1

thighL = arm.edit_bones['thigh.L']
thighL.tail.x = 0.2
thighL.tail.y = 0.09

shinL = arm.edit_bones['shin.L']
shinL.tail.x = 0.33
shinL.tail.y = 0.1

footL = arm.edit_bones['foot.L']
footL.tail.x = 0.33
footL.tail.y = 0.12

toeL = arm.edit_bones['toe.L']
toeL.tail.x = 0.33
toeL.tail.y = -0.12

heelL = arm.edit_bones['heel.02.L']
heelL.tail.x = 0.35
heelL.tail.y = 0.15
heelL.head.x = 0.33
heelL.head.x = 0.33

bpy.ops.object.mode_set(mode='OBJECT')

        #automatic weights parenting
 

