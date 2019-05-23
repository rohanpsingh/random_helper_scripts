import numpy as np
import os
import csv
import shutil
import glob
from tf import transformations
import cv2
from cv_bridge import CvBridge, CvBridgeError

rootPath = "/home/rohan/tmp/datasets/multiview_dataset/webcam2/images1"
rotationDir = "extrinsic"
translationDir = "extrinsic_t"
imagesDir = "images_use"

#create dirs if not exist (WARNING: data will be overwritten if dirs exist)
outDir = "out"
imageOutDir = os.path.join(rootPath, outDir, "image")
posesOutDir = os.path.join(rootPath, outDir, "poses")
if not os.path.isdir(imageOutDir):    os.makedirs(imageOutDir)
if not os.path.isdir(posesOutDir):    os.makedirs(posesOutDir)

for i in range(1, 1001):
    imgName = 'img' + repr(i).zfill(4)

    f = open(os.path.join(rootPath, rotationDir,imgName + '.txt'), 'r')
    lines = [line.rstrip("\n") for line in f.readlines()]
    line = lines[0]
    rot = [float(r) for r in line.split(",")]
    rot = np.asarray(rot).reshape(3,-1)

    f = open(os.path.join(rootPath, translationDir,imgName + '.txt'), 'r')
    lines = [line.rstrip("\n") for line in f.readlines()]
    line = lines[0]
    trns = [float(t) for t in line.split(",")]
    trns = np.asarray(trns)

    tf = np.eye(4,4)
    tf[:3,:3] = rot
    tf[:3,3] = trns
    #tf = np.linalg.inv(tf)

    posefile = os.path.join(posesOutDir, 'frame_' + repr(i).zfill(4) + '.txt')
    np.savetxt(posefile, tf)
    
    imgfile = os.path.join(imageOutDir, 'frame_' + repr(i).zfill(4) + '.jpg')
    shutil.copy(os.path.join(rootPath, imagesDir, imgName + '.jpg'), imgfile)
