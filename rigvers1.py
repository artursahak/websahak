bl_info = {
    "name": "Artur Rig AI",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy

import cv2
import dlib
import numpy as np
#print("dlib-",dlib.__version__,"cv2-",cv2.__version__,"numpy-",np.__version__)
import random
import mathutils



        
class ObjectMoveX(bpy.types.Operator):
    """My Object Moving Script"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.move_x"        # Unique identifier for buttons and menu items to reference.
    bl_label = "Artur Rig AI"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    def execute(self, context):        # execute() is called when running the operator.
        # OPengl snapshot
        
        #bpy.ops.render.opengl()
        #bpy.data.images["Render Result"].save_render("C:/wheels/im_kerneli_render.png")
        
        
        bpy.context.scene.render.filepath = 'C:/wheels/im_kerneli_render.png'
        bpy.ops.render.render(write_still = True)
        
        input_image = cv2.imread('C:\wheels\im_kerneli_render.png')
        img = input_image.copy()
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        fast = cv2.FastFeatureDetector_create()

        orb = cv2.ORB_create()

        keypoints = orb.detect(gray_image, None)

        keypoints, descriptors = orb.compute(gray_image, keypoints)

        cv2.drawKeypoints(input_image, keypoints, input_image, color=(0,255,0))


        canny = cv2.Canny(img, 50, 240)

        MODE = "COCO"

        if MODE is "COCO":
            protoFile = "C:/wheels/pose_deploy_linevec.prototxt"
            weightsFile = "C:/wheels/pose_iter_440000.caffemodel"
            nPoints = 18
            
            POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
        
        
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        
        #POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
        frame = cv2.imread("C:/wheels/im_kerneli_render.png")
        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        threshold = 0.1

        inWidth = 1280
        inHeight = 720
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]

        #guyner

        # Empty list to store the detected keypoints
        points = []
        converter = []
        #trained sheet xyz coords
        #A 0 || -1 0 87
        #B 1 || -1 0 74
        #A 1 || -1 0 74
        #B 2 || -13 0 67
        #A 2 || -13 0 67
        #B 3 || -21 0 58
        #A 3 || -21 0 58
        #B 4 || -21 0 48
        #A 1 || -1 0 74
        #B 5 || 9 0 69
        #A 5 || 9 0 69
        #B 6 || 21 0 58
        #A 6 || 21 0 58
        #B 7 || 21 0 48
        #A 1 || -1 0 74
        #B 14 || -1 0 54
        #A 14 || -1 0 54
        #B 8 || -5 0 39
        #A 8 || -5 0 39
        #B 9 || -9 0 20
        #A 9 || -9 0 20
        #B 10 || -13 0 0
        #A 14 || -1 0 54
        #B 11 || 5 0 39
        #A 11 || 5 0 39
        #B 12 || 9 0 20
        #A 12 || 9 0 20
        #B 13 || 9 0 0
        #fix = [0,1,2,3,1,5,6,1,14,8,9,14,11,12,13,14,15,16,17,18]
        fix = [2,2,5,5,2,2,5,5,2,2,2,2,2,2,2,2,2,2]
        
        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W 
            y = (frameHeight * point[1]) / H 
            #hashvarkner , calculations
            hasher = range(-7, 7)
            new_x = 0.15*(frameWidth*point[0])/W - 92
            new_z = -0.15*((frameHeight*point[1])/H) + 100
            
            numArr = [i]
            
            if prob > threshold : 
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(numArr), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                #bpy.ops.object.empty_add(type='CUBE', location=(int(new_x),0,int(new_z)))

                #bpy.ops.object.empty_add(type='CUBE', radius=12, view_align=False, location=(new_x, 0, new_z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
                converter.append((int(new_x),fix[i],int(new_z)))
                #bpy.ops.object.armature_add(view_align=False, enter_editmode=False, location=(new_x ,0, new_z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
                #bpy.ops.mesh.primitive_uv_sphere_add(radius=12, view_align=False, enter_editmode=False, location=(new_x, 0, new_z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
                #bpy.ops.mesh.primitive_cube_add(size=2.5, enter_editmode=False, location=(new_x, 0, new_z))
                #bpy.ops.object.armature_add(enter_editmode=False, location=(new_x, 0, new_z))
                #print(converter)
                
              

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else :
                points.append(None)
                
       
        
        
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0,255,0), 2)
                cv2.circle(frame, points[partA], 8, (0, 155, 20), thickness=-1)
                print(points[partA], points[partB])

        #print(chain, "here we go")
        obj = bpy.context.active_object
        #new type of bones 
        amt = bpy.data.armatures.new(obj.name + "_vBones")
        rig = bpy.data.objects.new(obj.name + '_vRig', amt)
        bpy.context.collection.objects.link(rig)
        bpy.context.view_layer.objects.active = rig
        bpy.context.view_layer.update()
        
        # Draw Skeleton
        bpy.ops.object.editmode_toggle()
        #for i in range(0, len(points) - 1):
            
                #bone = amt.edit_bones.new(str(i + 1))
                #bone.head = (converter[i][0],0,converter[i][1])
                #bone.tail = (converter[i+1][0],0,converter[i+1][1])
        
        #fixer = [1,1,4,1.7,1,1,3.5,1,2,2,1,2,2,1,1]
        #fixerB = [1,3,1,1,1,2,1,1,2,1,1,2,1,1,1]
        
        #skeleton = ["spine.006","spine","shoulder.R","upper_arm.R","fore_arm.R","shoulder.L","upper_arm.L","forearm.L","thigh.R",
        #"shin.R","shin.R","thigh.L","shin.L","shin.L","pelvis"]
      
        
        for k in POSE_PAIRS:
            boardA = k[0]
            boardB = k[1]
            y = 2
           
            bone = amt.edit_bones.new(str(boardA))
            bone.head = (converter[boardA][0],converter[boardA][1],converter[boardA][2])
            bone.tail = (converter[boardB][0],converter[boardA][1],converter[boardB][2])
            
            print("A",boardA,"||",converter[boardA][0],0,converter[boardA][1])
            #print("B",boardB,"||",converter[boardB][0],0,converter[boardB][1])
           
        
        for i in range(0, len(amt.edit_bones)-1 ):
           
                #amt.edit_bones[i+1].parent = amt.edit_bones[i] 
                amt.edit_bones[i+1].use_connect = True
        
        #for bone in arm.edit_bones:
            #if fnmatch.fnmatchcase(bone.name, "1.001"): 
                #arm.edit_bones.remove(bone)
        #    if fnmatch.fnmatchcase(bone.name, "1.002"): 
        #        arm.edit_bones.remove(bone)
        #    if fnmatch.fnmatchcase(bone.name, "14.001"): 
        #        arm.edit_bones.remove(bone) 
        
        
        bpy.ops.object.editmode_toggle()
        # lineType=cv2.FILLED
        #cv2.imshow('Output-Keypoints', frameCopy)
        #cv2.imshow('Output-Skeleton', frame)
        #cv2.imshow('BRIEF keyponts', input_image)
       
        
        cv2.imwrite('C:\wheels\Output-Keypoints.jpg', frameCopy)
        cv2.imwrite('C:\wheels\Output-Skeleton.jpg', frame)
        cv2.imwrite('C:\wheels\orb-keypoints.jpg',input_image)
        cv2.imwrite('C:\wheels\output-edges.jpg',canny)

        cv2.waitKey(0)

        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

def register():
    bpy.utils.register_class(ObjectMoveX)


def unregister():
    bpy.utils.unregister_class(ObjectMoveX)


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()