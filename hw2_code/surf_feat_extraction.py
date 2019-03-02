#!/usr/bin/env python3

import os
import sys
import threading
import yaml
import pickle
import pdb
import pandas as pd
import numpy as np
import cv2


def get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval, hessian_threshold):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."

    print("[INFO] get_surf_features_from_video: ", downsampled_video_filename)

    # Get frames from the video
    img_list = get_keyframes(downsampled_video_filename, keyframe_interval)

    # Initialize first row of feature list
    feature_list = np.zeros([1,64])

    for i in img_list:
        # Get gray scale of image
        image_g = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)

        # Create SURF object
        surf = cv2.xfeatures2d.SURF_create(hessian_threshold)

        # Obtain the descriptor
        kp, des = surf.detectAndCompute(image_g,None)

        if des is not None:
            # Concatenate with the existing list of descriptors
            feature_list = np.concatenate((feature_list, des), axis=0)

    # Flatten the array of 2D matrices into an array of arrays
    feature_list_flat = feature_list.flatten()

    # Delete the first dummy row added
    feature_list = np.delete(feature_list, (0), axis=0)

    # Save the final 2D matrix for the video
    df = pd.DataFrame(feature_list)
    df.to_csv(surf_feat_video_filename, index=False)


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


if __name__ == '__main__':
    print("Number of arguments", sys.argv)
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    hessian_threshold = my_params.get('hessian_threshold')
    surf_features_folderpath = my_params.get('surf_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)

    fread = open(all_video_names, "r")
    cnt = 1
    for line in fread.readlines():
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')

        if not os.path.isfile(downsampled_video_filename):
            continue

        # Get SURF features for one video
        print("File count", cnt)
        cnt += 1
        get_surf_features_from_video(downsampled_video_filename,
                                     surf_feat_video_filename, keyframe_interval,
                                     hessian_threshold)

