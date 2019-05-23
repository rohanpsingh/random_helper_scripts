import numpy as np
import os
import csv
import shutil
import cv2
from itertools import chain

#input data
rootPath = "/home/rohan/tmp/datasets/multiview_dataset/webcam2/images5"
tfcsv = "images5.csv"
imagesDir = "images"
camera_matrix = np.float32([[506.7917175292969, 0.0, 312.5103261144941], [0.0, 509.2976684570312, 231.8653046070103], [0.0, 0.0, 1.0]])

#dims of the turntable (from marker-center to marker-center)
dim_x = 0.358
dim_y = 0.227

#output dirs
rotationDir = "extrinsic"
translationDir = "extrinsic_t"
imagesuseDir = "images_use"

#create dirs if not exist (WARNING: data will be overwritten if dirs exist)
if not os.path.isdir(os.path.join(rootPath, rotationDir)):
    os.makedirs(os.path.join(rootPath, rotationDir))
if not os.path.isdir(os.path.join(rootPath, translationDir)):    
    os.makedirs(os.path.join(rootPath, translationDir))
if not os.path.isdir(os.path.join(rootPath, imagesuseDir)):
    os.makedirs(os.path.join(rootPath, imagesuseDir))

#open the csv file (this file is the output of the AIST lenti marker program) 
f = open(os.path.join(rootPath, tfcsv), 'r')
lines = [line.rstrip("\n") for line in f.readlines()]
lines = lines[1:]

rot_1 = np.zeros([3,3])
trns_1 = np.zeros([3])
rot_2 = np.zeros([3,3])
trns_2 = np.zeros([3])
rot_3 = np.zeros([3,3])
trns_3 = np.zeros([3])
rot_4 = np.zeros([3,3])
trns_4 = np.zeros([3])

# transformtaions from each marker to the first marker
# origin is fixed to the center of '1st marker'
tf21 = np.eye(4)
tf21[:3,3] = np.array([dim_x,0,0])
tf31 = np.eye(4)
tf31[:3,3] = np.array([dim_x,dim_y,0])
tf41 = np.eye(4)
tf41[:3,3] = np.array([0,dim_y,0])

#counter for each valid datapoint
count = 0

for line in lines:
    id1 = False
    id2 = False
    id3 = False
    id4 = False
    
    l = [r for r in line.split(",")]
    imgfile = os.path.join(rootPath, imagesDir, l[0] + '.jpg')
    
    l = l[2:162]

    #select the data chunk for marker
    l_1 = l[90:102]
    l_2 = l[105:117]
    l_3 = l[120:132]
    l_4 = l[135:147]

    #extract data from each line and rearrange
    for i in range(3):
        rot_1[i] = l_1[(i*4):(i*4)+3]
        trns_1[i] = l_1[(i+1)*4 - 1]
    for i in range(3):
        rot_2[i] = l_2[(i*4):(i*4)+3]
        trns_2[i] = l_2[(i+1)*4 - 1]
    for i in range(3):
        rot_3[i] = l_3[(i*4):(i*4)+3]
        trns_3[i] = l_3[(i+1)*4 - 1]
    for i in range(3):
        rot_4[i] = l_4[(i*4):(i*4)+3]
        trns_4[i] = l_4[(i+1)*4 - 1]

    #set flags for visible markers
    if float(l[102])!=0.0:       id1 = True
    if float(l[117])!=0.0:       id2 = True
    if float(l[132])!=0.0:       id3 = True
    if float(l[147])!=0.0:       id4 = True

    #if no marker is visible 
    if not id1 and not id2 and not id3 and not id4:
        continue

    #init
    tf = np.eye(4)
    
    #average rotations from all visible markers
    tf[:3,:3] = (id1*rot_1 + id2*rot_2 + id3*rot_3 + id4*rot_4)/float(id1 + id2 + id3 + id4)
    r = tf[:3,:3]

    #translations need to be in the same frame for avg to make sense
    if id1:
        tf[:3,3] = trns_1
        trns_1 = np.dot(tf, np.linalg.inv(np.eye(4)))[:3,3]
    if id2:
        tf[:3,3] = trns_2
        trns_2 = np.dot(tf, np.linalg.inv(tf21))[:3,3]
    if id3:
        tf[:3,3] = trns_3
        trns_3 = np.dot(tf, np.linalg.inv(tf31))[:3,3]
    if id4:
        tf[:3,3] = trns_4
        trns_4 = np.dot(tf, np.linalg.inv(tf41))[:3,3]

    #average translations from all visible markers
    tf[:3,3] = (id1*trns_1 + id2*trns_2 + id3*trns_3 + id4*trns_4)/float(id1 + id2 + id3 + id4)
    
    if np.isnan(tf).any():
        continue
    else:
        count = count + 1

    #write data to appropriate directories
    filename = 'img' + repr(count).zfill(4)
    f = open(os.path.join(rootPath, rotationDir, filename + '.txt'), 'w')
    wr = csv.writer(f, dialect='excel')
    wr.writerow(list(chain.from_iterable(tf[:3,:3])))

    f = open(os.path.join(rootPath, translationDir, filename + '.txt'), 'w')
    wr = csv.writer(f, dialect='excel')
    wr.writerow(list(tf[:3,3]))

    shutil.copy(imgfile, os.path.join(rootPath, imagesuseDir, filename + '.jpg'))
