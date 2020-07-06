import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers.merge import Concatenate
from utils import *
import json
import pandas

weights_path = "model.h5" # orginal weights converted from caffe
input_shape = (None,None,3)

img_input = Input(shape=input_shape)

stages = 6
np_branch1 = 38
np_branch2 = 19

img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

# VGG
stage0_out = vgg_block(img_normalized)

# stage 1
stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1)
stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2)
x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

# stage t >= 2
for sn in range(2, stages + 1):
    stageT_branch1_out = stageT_block(x, np_branch1, sn, 1)
    stageT_branch2_out = stageT_block(x, np_branch2, sn, 2)
    if (sn < stages):
        x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
model.load_weights(weights_path)

# body_parts = {"Nose" : 0, "Neck" : 1, "Right Shoulder" : 2, "Right Elbow" : 3, "Right Wrist" : 4, "Left Shoulder" : 5,
#               "Left Elbow" : 6, "Left Wrist" : 7, "Right Hip" : 8, "Right Knee" : 9, "Right Ankle" : 10,
#               "Left Hip" : 11, "Left Knee" : 12, "LAnkle" : 13, "Right Eye" : 14, "Left Eye" : 15, "Right Ear" : 16,
#               "Left Ear" : 17}

param = {'use_gpu': 1, 'GPUdeviceNumber': 0, 'modelID': '1', 'octave': 3, 'starting_range': 0.8, 'ending_range': 2.0, 'scale_search': [0.5, 1.0, 1.5, 2.0], 'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5, 'min_num': 4, 'mid_num': 10, 'crop_ratio': 2.5, 'bbox_ratio': 0.25}
model_params = {'caffemodel': './model/_trained_COCO/pose_iter_440000.caffemodel', 'deployFile': './model/_trained_COCO/pose_deploy.prototxt', 'description': 'COCO Pose56 Two-level Linevec', 'boxsize': 368, 'padValue': 128, 'np': '12', 'stride': 8, 'part_str': ['[nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear', 'pt19]']}
#imgs = ["front","left","right"]
imgs = ["garik_front","garik_left","garik_right"]
dic = calculate_points(imgs, model,param,model_params)

coords_x = []
coords_y = []
coords_z = []
mypath = "C:/wheels/"
def coords(x,y,z):              
    ff = pandas.DataFrame(data={"x":x,"y":y,"z":z})
    ff.to_csv(mypath+"coords2.csv",sep=',',index=False)



body_parts = {"Nose" : 0, "Neck" : 1, "Right Shoulder" : 2, "Right Elbow" : 3, "Right Wrist" : 4, "Left Shoulder" : 5,
              "Left Elbow" : 6, "Left Wrist" : 7, "Right Hip" : 8, "Right Knee" : 9, "Right Ankle" : 10,
              "Left Hip" : 11, "Left Knee" : 12, "LAnkle" : 13}

alpha_scale = 0.005
dic_parts = {}
detect = []
for i in body_parts:
    if "Right" in i:
        detect.append(body_parts[i])
        print(i,"------",len(detect))
        front = dic[imgs[0]][body_parts[i]][0][0:2]
        print("Dictionary",dic[imgs[2]][body_parts[i]])
        right = dic[imgs[2]][body_parts[i]][0][0:2]
        x = (front[0]-600) * alpha_scale
        z = (315 - np.round(np.mean([front[1], right[1]]))) * alpha_scale
        y = (right[0]-600) * alpha_scale

        coords_x.append(x)
        coords_y.append(y)
        coords_z.append(z)
        dic_parts[i] = [x,z,y]
    elif "Left" in i:
        front = dic[imgs[0]][body_parts[i]][0][0:2]
        left = dic[imgs[1]][body_parts[i]][0][0:2]
        x = (front[0]-600) * alpha_scale
        z = (315-np.round(np.mean([front[1], left[1]]))) * alpha_scale
        y = ((1200-left[0])-600) * alpha_scale
        coords_x.append(x)
        coords_y.append(y)
        coords_z.append(z)
        dic_parts[i] = [x,z,y]
    else:
        front = dic[imgs[0]][body_parts[i]][0][0:2]
        right = dic[imgs[2]][body_parts[i]][0][0:2]
        left = dic[imgs[1]][body_parts[i]][0][0:2]
        x = (front[0]-600) * alpha_scale
        z = (315 - np.round(np.mean([front[1], right[1],left[1]]))) * alpha_scale
        y = (np.mean([right[0], (1200-left[0])])-600) * alpha_scale
        coords_x.append(x)
        coords_y.append(y)
        coords_z.append(z)
        dic_parts[i] = [x,z,y]

coords(coords_x,coords_y,coords_z)

