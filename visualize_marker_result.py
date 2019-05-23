import numpy as np
import os
import csv
import shutil
import glob
from tf import transformations
import cv2
from cv_bridge import CvBridge, CvBridgeError

#rootPath = "/home/rohan/tmp/multiview_dataset"
rootPath = "/home/rohan/tmp/datasets/multiview_dataset/webcam2/images1"
rotationDir = "extrinsic"
translationDir = "extrinsic_t"
imagesDir = "images_use"
#camera_matrix = np.float32([[834.07950, 0.0, 326.48264], [0.0, 831.62448, 244.12608], [0.0, 0.0, 1.0]])
camera_matrix = np.float32([[506.7917175292969, 0.0, 312.5103261144941], [0.0, 509.2976684570312, 231.8653046070103], [0.0, 0.0, 1.0]])


p1 = np.array([224,256,1])
p2 = np.array([224,256,1])
p3 = np.array([224,256,1])
p4 = np.array([224,256,1])

kps0 = np.float32([])
kps1 = np.float32([])
kps2 = np.float32([])
for i in range(100):    kps0 = np.append(kps0, np.float32([0.0, i*0.002, 0.0]))
for i in range(100):    kps1 = np.append(kps1, np.float32([i*0.002, 0.0, 0.0]))
for i in range(100):    kps2 = np.append(kps2, np.float32([0.0, 0.0, i*0.002]))
kps0 = kps0.reshape(-1,3)
kps1 = kps1.reshape(-1,3)
kps2 = kps2.reshape(-1,3)

for i in range(1, 1001):
    print "-----", i
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

    imgfile = os.path.join(rootPath, imagesDir, imgName + '.jpg')
    img = cv2.imread(imgfile)
    for p in kps0:
        kp = np.dot(tf[:3,:3], p) + tf[:3,3]
        uv = np.dot(camera_matrix, kp)
        uv = uv/uv[2]
        cv2.circle(img, (int(uv[0]), int(uv[1])), 5, (0,0,255), -1)

    for p in kps1:
        kp = np.dot(tf[:3,:3], p) + tf[:3,3]
        uv = np.dot(camera_matrix, kp)
        uv = uv/uv[2]
        cv2.circle(img, (int(uv[0]), int(uv[1])), 5, (0,255,0), -1)

    for p in kps2:
        kp = np.dot(tf[:3,:3], p) + tf[:3,3]
        uv = np.dot(camera_matrix, kp)
        uv = uv/uv[2]
        cv2.circle(img, (int(uv[0]), int(uv[1])), 5, (255,0,0), -1)
        
    cv2.imshow("win", img)
    cv2.waitKey(50)
