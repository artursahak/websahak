import cv2
import time
import numpy as np
import bpy

MODE = "COCO"

if MODE is "COCO":
    protoFile = "C:/wheels/pose_deploy_linevec.prototxt"
    weightsFile = "C:/wheels/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]


net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


bpy.context.scene.render.filepath = 'C:/wheels/im_kerneli_render.png'
bpy.ops.render.render(write_still = True)

frame = cv2.imread("C:/wheels/im_kerneli_render.png")
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

    if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        converter.append((int(new_x),int(new_z)))
        # Add the point to the list if the probability is greater than the threshold
        bpy.ops.object.empty_add(type='CUBE', location=(int(new_x),0,int(new_z)))
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
        
  
      

            #print("B",boardB,"||",converter[boardB][0],0,converter[boardB][1])
        
        
cv2.imwrite('C:/wheels/Output-Keypoints2.jpg', frameCopy)
cv2.imwrite('C:/wheels/Output-Skeleton2.jpg', frame)


        


#print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)

