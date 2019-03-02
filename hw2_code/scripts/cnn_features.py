import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import sys
import threading
import yaml
import pickle
import pdb
import pandas as pd
import numpy as np
import cv2

from PIL import Image
import PIL

# Load the pretrained model
model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_g):
    # 1. Convert image to PIL format
    img = PIL.Image.fromarray(image_g, mode=None)

    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)

    # Get frames one by one
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img

    # Release video object
    video_cap.release()

def get_cnn_features_from_video(downsampled_video_filename, cnn_feat_video_filename, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."

    print("[INFO] get_cnn_features_from_video: ", downsampled_video_filename)

    # Get frames from the video
    img_list = get_keyframes(downsampled_video_filename, keyframe_interval)

    # Initialize first row of feature list
    feature_list = np.zeros([1,512])
    
    print(feature_list.shape)

    for i in img_list:
        temp = np.matrix(get_vector(i))

        feature_list = np.concatenate((feature_list, temp), axis=0)

    # Flatten the array of 2D matrices into an array of arrays
    #feature_list_flat = feature_list.flatten()

    # Delete the first dummy row added
    feature_list = np.delete(feature_list, (0), axis=0)

    # Save the final 2D matrix for the video
    df = pd.DataFrame(feature_list)
    df.to_csv(cnn_feat_video_filename, index=False)


all_video_names = "list/all.video"
config_file = "config.yaml"
my_params = yaml.load(open(config_file))

# Get parameters from config file
keyframe_interval = my_params.get('keyframe_interval')
hessian_threshold = my_params.get('hessian_threshold')
cnn_features_folderpath = my_params.get('cnn_features')
downsampled_videos = my_params.get('downsampled_videos')

# Loop over all videos (training, val, testing)

fread = open(all_video_names, "r")
cnt = 1
for line in fread.readlines():
    video_name = line.replace('\n', '')
    downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
    cnn_feat_video_filename = os.path.join(cnn_features_folderpath, video_name + '.cnn')

    if not os.path.isfile(downsampled_video_filename):
        continue

    # Get cnn features for one video
    print("File count", cnt)
    cnt += 1
    get_cnn_features_from_video(downsampled_video_filename,
                                 cnn_feat_video_filename, keyframe_interval)


