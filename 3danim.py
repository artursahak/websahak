import bpy
import cv2
import numpy as np
import random
import mathutils
import time

images = ["/garik.png","/garik2.png","/garik3.png"]

MODE = "COCO"
path = "/home/artur/Documents/coords"
if MODE is "COCO":
    protoFile = path+"/pose_deploy_linevec.prototxt"
    weightsFile = path+"/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

frame = cv2.imread(path+images[0])
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

t = time.time()
# input image dimensions for the network
height, width, channels = frame.shape 
print(frame.shape)
inWidth = width
inHeight = height
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),(0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

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
  
    new_x = 0.15*(x) - 93
    new_z = -0.15*(y) + 93
    
    numArr = [i]
    
    if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(numArr), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        converter.append((int(new_x),int(new_z)))

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


cv2.imwrite(path+'/Output-Skeleton.jpg', frame)
cv2.imwrite(path+'/Output-Keypoints.jpg', frameCopy)



frame = cv2.imread(path+images[1])
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

t = time.time()
# input image dimensions for the network
height, width, channels = frame.shape 
print(frame.shape)
inWidth = width
inHeight = height
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),(0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]
#guyner

# Empty list to store the detected keypoints
points = []
converter_left = []


for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    
    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W 
    y = (frameHeight * point[1]) / H 
    #hashvarkner , calculations
  
    new_x = 0.1*(x) - 65
    new_z = -0.1*(y)
    
    numArr = [i]
    
    if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(numArr), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        converter_left.append((int(-new_x),int(new_z)))

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


cv2.imwrite(path+'/Output-Skeleton2.jpg', frame)
cv2.imwrite(path+'/Output-Keypoints2.jpg', frameCopy)


frame = cv2.imread(path+images[2])
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

t = time.time()
# input image dimensions for the network
height, width, channels = frame.shape 
print(frame.shape)
inWidth = width
inHeight = height
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),(0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]
#guyner

# Empty list to store the detected keypoints
points = []
converter_right = []


for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    
    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W 
    y = (frameHeight * point[1]) / H 
    #hashvarkner , calculations
  
    new_x = 0.1*(x) - 65
    new_z = -0.1*(y) 
    
    numArr = [i]
    
    if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(numArr), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        converter_right.append((int(-new_x),int(new_z)))

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


cv2.imwrite(path+'/Output-Skeleton3.jpg', frame)
cv2.imwrite(path+'/Output-Keypoints3.jpg', frameCopy)


converter_3d = converter_left[:7] + converter_right[:11]


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
array = []        
 
for k in POSE_PAIRS:
    boardA = k[0]
    boardB = k[1]
    y = 2
    array.append(str(k))
    
    bone = amt.edit_bones.new(str(len(array)))
    bone.head = (converter[boardA][0],converter_3d[boardA][0],converter[boardA][1])
    bone.tail = (converter[boardB][0],converter_3d[boardB][0],converter[boardB][1])
    
    print(len(k))
    
    #print("A",boardA,"||",converter[boardA][0],0,converter[boardA][1])
    #print("B",boardB,"||",converter[boardB][0],0,converter[boardB][1])
list1 = [1,2,3,4,0]
print(range(0,len(list1)))
#for i in range(0,len(list1)-1):
    #print(list1[i])
    #amt.edit_bones[list1[i]].parent = amt.edit_bones[list1[i+1]] 

for i in range(0, len(amt.edit_bones)-1):
   
        #amt.edit_bones[i+1].parent = amt.edit_bones[i] 
        amt.edit_bones[i+1].use_connect = True

# parent Bones




amt.edit_bones["4"].parent = amt.edit_bones["2"]
amt.edit_bones["5"].parent = amt.edit_bones["4"]
amt.edit_bones["6"].parent = amt.edit_bones["3"]
amt.edit_bones["7"].parent = amt.edit_bones["6"]
amt.edit_bones["9"].parent = amt.edit_bones["8"]
amt.edit_bones["10"].parent = amt.edit_bones["9"]
amt.edit_bones["12"].parent = amt.edit_bones["11"]
amt.edit_bones["13"].parent = amt.edit_bones["12"]
#amt.edit_bones["8"].parent = amt.edit_bones["2"]
#amt.edit_bones["11"].parent = amt.edit_bones["3"]
#amt.edit_bones["8"].parent = amt.edit_bones["1"]
#amt.edit_bones["11"].parent = amt.edit_bones["1"]
bpy.ops.object.editmode_toggle()


bpy.ops.object.mode_set(mode="OBJECT")

objects = bpy.data.objects
objects['garik'].select_set(True)
objects['garik_vRig'].select_set(True)