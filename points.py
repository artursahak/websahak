import bpy
import json
import mathutils
path = "/home/artur/Documents/coords/"
with open(path+"blender_points.json","r") as f:
    blender_points = json.load(f)
    
def initialize_metarig():
    bpy.ops.object.armature_basic_human_metarig_add()
    bpy.context.object.data.show_names = True

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
x = []
y = []
z = []

for i in blender_points:
    new_x = blender_points[i][0]
    new_z = blender_points[i][1]
    new_y = blender_points[i][2]
    x.append(new_x)
    y.append(new_y)
    z.append(new_z)    
    


upper_armL=arm.edit_bones['upper_arm.L']
upper_armL.tail.x = x[6]
upper_armL.tail.y = y[6]
upper_armL.tail.z = z[6]

fore_armL = arm.edit_bones['forearm.L']
fore_armL.tail.x = x[7]
fore_armL.tail.y = y[7]
fore_armL.tail.z = z[7]

handL = arm.edit_bones['hand.L']
handL.tail.x = x[7]
handL.tail.y = y[7]
handL.tail.z = z[7]+0.1

thighL = arm.edit_bones['thigh.L']
thighL.tail.x = x[12]
thighL.tail.y = y[12]

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


#upperarmX = converter[6][0]
#upperarmZ = converter[6][1]
bpy.ops.object.mode_set(mode='OBJECT')
