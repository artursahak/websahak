import bpy
import cv2
import dlib
import numpy as np
#print("dlib-",dlib.__version__,"cv2-",cv2.__version__,"numpy-",np.__version__)
import random

class SimpleOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.simple_operator"
    bl_label = "Artur AI rig"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        
        bpy.ops.render.opengl()
        bpy.data.images["Render Result"].save_render("C:/wheels/im_kerneli_render.png")

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
            new_x = 0.1*(frameWidth*point[0])/W
            new_z = -0.1*((frameHeight*point[1])/H)
            
            numArr = [new_x,new_z]
            
            if prob > threshold : 
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(numArr), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                print(numArr)
                #bpy.ops.object.empty_add(type='CUBE', radius=12, view_align=False, location=(new_x, 0, new_z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
                
                #bpy.ops.object.armature_add(view_align=False, enter_editmode=False, location=(new_x ,0, new_z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
                #bpy.ops.mesh.primitive_uv_sphere_add(radius=12, view_align=False, enter_editmode=False, location=(new_x, 0, new_z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
                bpy.ops.mesh.primitive_cube_add(size=5, enter_editmode=False, location=(new_x, 0, new_z))
                #bpy.ops.object.armature_add(enter_editmode=False, location=(new_x, 0, new_z))


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


        cv2.imwrite('C:\wheels\Output-Keypoints.jpg', frameCopy)
        cv2.imwrite('C:\wheels\Output-Skeleton.jpg', frame)
        cv2.imwrite('C:\wheels\orb-keypoints.jpg',input_image)
        cv2.imwrite('C:\wheels\output-edges.jpg',canny)

        cv2.waitKey(0)
        

        return {'FINISHED'}


def draw_func(self, context):
    layout = self.layout
    layout.operator("object.simple_operator")
    

def register():
    bpy.utils.register_class(SimpleOperator)
    bpy.types.VIEW3D_HT_header.prepend(draw_func)


def unregister():
    bpy.utils.unregister_class(SimpleOperator)
    bpy.types.VIEW3D_HT_header.remove(draw_func)


if __name__ == "__main__":
    register()
