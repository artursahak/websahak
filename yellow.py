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

        protoFile = "C:/wheels/pose_deploy_linevec_faster_4_stages.prototxt";
        weightsFile = "C:/wheels/pose_iter_160000.caffemodel";
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        nPoints = 15
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
        frame = cv2.imread("C:/wheels/im_kerneli_render.png")
        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        threshold = 0.1

        inWidth = 400
        inHeight = 400
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]

        #guyner

        # Empty list to store the detected keypoints
        points = []
        converter = []
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
            new_x = 0.1*(frameWidth*point[0])/W - 94
            new_z = -0.1*((frameHeight*point[1])/H) + 100
            
            numArr = [i,":","x",new_x,"|z",new_z]
            
            if prob > threshold : 
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(numArr), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                
                #bpy.ops.object.empty_add(type='CUBE', radius=12, view_align=False, location=(new_x, 0, new_z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
                converter.append((int(new_x),int(new_z)))
                #bpy.ops.object.armature_add(view_align=False, enter_editmode=False, location=(new_x ,0, new_z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
                #bpy.ops.mesh.primitive_uv_sphere_add(radius=12, view_align=False, enter_editmode=False, location=(new_x, 0, new_z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
                #bpy.ops.mesh.primitive_cube_add(size=2.5, enter_editmode=False, location=(new_x, 0, new_z))
                #bpy.ops.object.armature_add(enter_editmode=False, location=(new_x, 0, new_z))
                #print(converter)
                
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else :
                points.append(None)

        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0,255,0), 2)
                cv2.circle(frame, points[partA], 8, (0, 155, 20), thickness=-1)

        # lineType=cv2.FILLED
        #cv2.imshow('Output-Keypoints', frameCopy)
        #cv2.imshow('Output-Skeleton', frame)
        #cv2.imshow('BRIEF keyponts', input_image)
        def initialize_metarig():
            bpy.ops.object.armature_basic_human_metarig_add()
            bpy.ops.transform.resize(value=(45.7559, 45.7559, 45.7559), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    #bpy.ops.object.posemode_toggle()
    #bpy.context.object.pose.use_mirror_x = True
    #bpy.ops.pose.group_deselect()
#handL.tail=(17.36,0,52.48)
        initialize_metarig()
        
        #start controlling the bones !! -- very important mas --! hayeren tekst : 
        bpy.data.objects["metarig"].select_set(True)
        
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.context.object.data.use_mirror_x = True
        bpy.ops.armature.select_all(action='DESELECT')
        
        ob = bpy.data.objects["metarig"]
        arm = ob.data
        #mod = 60
        
        
        upperarmX = converter[5][0]
        upperarmZ = converter[5][1]
        
        #upper_armL=arm.edit_bones['upper_arm.L']
        #upper_armL.tail=(-upperarmX/mod,0,upperarmZ/mod)
        #upper_armL.tail.x += 0.2
        #bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(LarmtailX, 0, LarmtailZ))
        
        LhandheadX = converter[6][0]+10
        LhandheadZ = converter[6][1]
        
        LhandtailX = converter[6][0] 
        LhandtailZ = converter[6][1] 
        #print(LarmtailX,LarmtailZ,LhandheadX,LhandheadZ,LhandtailX,LhandtailZ)  
        
        
        handL = arm.edit_bones['hand.L']
        handL.tail.x -= 0.25
        handL.tail.z -= 0.25
        handL.head.x -= 0.25
        handL.head.z -= 0.15
        #handL.head = (LhandheadX/mod,0,LhandheadZ/mod)
        #handL.tail = (LhandheadX/mod,0,LhandheadZ/mod)
        #bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(LhandtailX, 0, LhandtailZ))
        #save the pics
        shinL = arm.edit_bones['shin.L']
        shinL.tail.x += 0.12
        shinL.head.x += 0.12
        shinL.head.y += 0.1
        
        #footL.head = (shinLX,0,shinLZ)
        #footL.tail = (shinLX_tail,0,shinLZ_tail)
        
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