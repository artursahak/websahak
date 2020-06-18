import cv2
import time
import numpy as np
import math
import bpy

MODE = "COCO"
path = "/home/artur/Documents/coords"
if MODE is "COCO":
    protoFile = path+"/pose_deploy_linevec.prototxt"
    weightsFile = path+"/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

frame = cv2.imread(path+"/im_kerneli_render.png")
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

t = time.time()
# input image dimensions for the network
inWidth = 1280
inHeight = 720
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),(0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]

# Empty list to store the detected keypoints
points = []
converter = []
pointX = []
pointY = []

for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H
    new_x = 0.1*(frameWidth*point[0])/W 
    new_z = -0.1*((frameHeight*point[1])/H) 
    pointX.append(x)
    pointY.append(y)
    if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        converter.append((int(new_x),int(new_z)))
        # Add the point to the list if the probability is greater than the threshold

        points.append((int(x), int(y)))
    else :
        points.append(None)
        
# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #for i in range(0, len(points) - 1):
            
                #bone = amt.edit_bones.new(str(i + 1))
                #bone.head = (converter[i][0],0,converter[i][1])
                #bone.tail = (converter[i+1][0],0,converter[i+1][1])
        
        #fixer = [1,1,4,1.7,1,1,3.5,1,2,2,1,2,2,1,1]
        #fixerB = [1,3,1,1,1,2,1,1,2,1,1,2,1,1,1]
#print(a,b ,"||",c,"tan:",arctan)     

            #print("B",boardB,"||",converter[boardB][0],0,converter[boardB][1])
        
        
#cv2.imwrite('/home/artur/Documents/coords/Output-Keypoints2.jpg', frameCopy)
#cv2.imwrite('/home/artur/Documents/coords/Output-Skeleton2.jpg', frame)

ob = bpy.data.objects['metarig']
bpy.context.view_layer.objects.active = ob
bpy.ops.object.mode_set(mode="POSE")


def move_left_bone(name,num1,num2,times):
    a = pointY[num1]-pointY[num2]
    b = pointX[num1]-pointY[num2]
    c = math.sqrt(a**2+b**2)
    tanz = -(b/c)
    arctan = math.atan(tanz)
    pbone = ob.pose.bones[str(name)] # "upper_arm.L"
    pbone.rotation_mode = 'XYZ'  # 
    axis = "Z"
    angle = times * arctan
    pbone.rotation_euler.rotate_axis(axis,angle)
    print(a,b ,"||",c,"tan:",arctan,"-",angle)
    
move_left_bone("upper_arm.L",6,5,0.3)
move_left_bone("forearm.L",7,6,0.5)
move_left_bone("thigh.L",12,11,0.2)


def move_right_bone(name,num1,num2,times):
    a = pointY[num1]-pointY[num2]
    b = pointX[num1]-pointY[num2]
    c = math.sqrt(a**2+b**2)
    tanz = (b/c)
    arctan = math.atan(tanz)
    pbone = ob.pose.bones[str(name)] # "upper_arm.L"
    pbone.rotation_mode = 'XYZ'  # 
    axis = "Z"
    angle = times * arctan
    pbone.rotation_euler.rotate_axis(axis,angle)
    print(a,b ,"||",c,"tan:",arctan,"-",angle)

move_right_bone("upper_arm.R",3,2,0.3)
move_right_bone("forearm.R",4,3,0.7)

move_right_bone("thigh.R",9,8,0.2)

bpy.ops.object.mode_set(mode="OBJECT")


#print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)

