import numpy as np
import os
import cv2

rootPath = "/home/rohan/tmp/datasets/multiview_dataset/webcam2/images1"
rotationDir = "extrinsic"
translationDir = "extrinsic_t"
imagesDir = "images_use"
#camera_matrix = np.float32([[834.07950, 0.0, 326.48264], [0.0, 831.62448, 244.12608], [0.0, 0.0, 1.0]])
camera_matrix = np.float32([[506.7917175292969, 0.0, 312.5103261144941], [0.0, 509.2976684570312, 231.8653046070103], [0.0, 0.0, 1.0]])

outDir = "out"
labelOutDir = os.path.join(rootPath, outDir, "label")
if not os.path.isdir(labelOutDir):    os.makedirs(labelOutDir)



x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False
getROI = False

def click_and_crop(event, x, y, flags, param):
	global x_start, y_start, x_end, y_end, cropping, getROI
	if event == cv2.EVENT_LBUTTONDOWN:
		x_start, y_start, x_end, y_end = x, y, x, y
		cropping = True
	elif event == cv2.EVENT_MOUSEMOVE:
		if cropping == True:
			x_end, y_end = x, y
	elif event == cv2.EVENT_LBUTTONUP:
		x_end, y_end = x, y
		cropping = False
		getROI = True

def selectKeypoints(image, tot_keypts, win):
    clone = image.copy()
    kp_count = 0
    kpts = []
    global x_start, y_start, x_end, y_end, cropping, getROI
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 3000,2000)
    cv2.setMouseCallback(win, click_and_crop)
    while True:
            i = image.copy()
            if not cropping and not getROI:
                    cv2.imshow(win, image)
            elif cropping and not getROI:
                    cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                    cv2.imshow(win, i)
            elif not cropping and getROI:
                    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                    cv2.imshow(win, image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):            # if the 'r' key is pressed
                    image = clone.copy()
                    getROI = False
            if key == ord("n"):            # if the 'n' key is pressed
                    kp_count += 1
                    image = clone.copy()
                    keyPt = [(x_end + x_start)/2, (y_end + y_start)/2]
                    kpts.append(keyPt) 
                    getROI = False
                    if kp_count == tot_keypts:
                        break
            if key == ord("s"):            # if the 's' key is pressed
                    kp_count += 1
                    image = clone.copy()
                    keyPt = [-1, -1]
                    kpts.append(keyPt)
                    getROI = False
                    if kp_count == tot_keypts:
                            break
            elif key == ord("q"):            # if the 'q' key is pressed, break from the loop
                    while kp_count<tot_keypts:
                            kp_count += 1
                            keyPt = [-1, -1]
                            kpts.append(keyPt)
                    break
    cv2.destroyAllWindows()
    return kpts


def nearest_intersection(points, dirs):
    dirs_mat = (dirs[:, :, np.newaxis])*(dirs[:, np.newaxis, :])
    points_mat = points[:, :, np.newaxis]
    I = np.eye(3)
    x = (I - dirs_mat).sum(axis=0)
    y = (((I - dirs_mat)*points_mat).sum(axis=1)).sum(axis=0)[:,None]
    #sol = np.linalg.lstsq(x.sum(axis=0), y.sum(axis=0))[0]
    sol = np.linalg.inv(x).dot(y)
    return sol
    
def getRays(x, rots):
    pts = np.zeros([x.shape[0], x.shape[1], 3])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not (x[i,j] == [-1, -1]).all():
                rgb_uv = np.array([x[i,j,0], x[i,j,1], 1])
                rgb_uv = np.dot(np.linalg.inv(camera_matrix), rgb_uv)
                rgb_uv = rgb_uv/np.linalg.norm(rgb_uv)
                pts[i][j] = np.dot(rots[i], rgb_uv)
    return pts

def getIntersectionPoints(x, trns):
    int_points  = np.zeros([x.shape[1], 3])
    for i in range(x.shape[1]):
        dirs = x[:,i,:]
        visible = ~np.all(dirs == 0, axis=1)
        if visible.sum() > 2:
            dirs = dirs[visible]
            dirs = dirs/np.linalg.norm(dirs, axis=1)[:,None]
            point = nearest_intersection(trns[visible], dirs)
            int_points[i] = point[:,0]
    mask = ~np.all(int_points == 0, axis=1)
    return int_points, mask

def procrustes(X, Y, scaling=True, reflection='best'):
    n,m = X.shape
    ny,my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform
    


projpoints = np.asarray([
    [-0.000658277, -0.0218518, 0.229076],
    [-0.00133976, -0.039011, 0.197079],
    [-0.0216983, -0.0235916, 0.075856],
    [0.00500683, -0.0335716, 0.0377131],
    [-0.0594902, -0.036074, 0.0125283],
    [0.0290488, -0.0360729, 0.0129409],
    [0.0385984, -0.000909687, 0.238414],
    [0.0427567, 0.000419152, 0.0361168],
    [0.0439732, -0.0179427, 0.0136495],
    [0.0438782, 0.0182322, 0.0141658],
    [0.0141765, 0.0167749, 0.229616],
    [0.00012472, 0.0389563, 0.196798],
    [-0.0211969, 0.0236049, 0.074683],
    [0.00699471, 0.0335475, 0.0368053],
    [0.0305733, 0.0360479, 0.0129379],
    [-0.0580753, 0.0360481, 0.0138757],
    [-0.0572322, 0.0145193, 0.0380522],
    [-0.0578135, -0.0149183, 0.0387937],
    [-0.069522, 0.0142143, 0.0269248],
    [-0.0689446, -0.0147463, 0.027574]])


rotations = []
translations = []
kpts = []
l = [2, 58, 426, 427, 439, 455, 479, 653, 685, 700, 753]
ids = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17]
objpoints = projpoints[ids]
num_kpts = len(ids)

if False:
    for i in l:
        imgName = 'img' + repr(i).zfill(4)
        img = cv2.imread(os.path.join(rootPath, imagesDir, imgName + '.jpg'))
        kpts.append(selectKeypoints(img, num_kpts, imgName))

        f = open(os.path.join(rootPath, rotationDir,imgName + '.txt'), 'r')
        lines = [line.rstrip("\n") for line in f.readlines()]
        line = lines[0]
        rot = [float(r) for r in line.split(",")]
        rotations.append(rot)

        f = open(os.path.join(rootPath, translationDir,imgName + '.txt'), 'r')
        lines = [line.rstrip("\n") for line in f.readlines()]
        line = lines[0]
        trns = [float(t) for t in line.split(",")]
        translations.append(trns)

    kpt_arr = np.asarray(kpts).reshape(-1, num_kpts, 2)
    rotations = np.asarray(rotations).reshape(-1,3,3)
    translations = np.asarray(translations).reshape(-1,3)
    transformations = np.zeros([rotations.shape[0], 4, 4])
    transformations[:,0:3,0:3] = rotations
    transformations[:,0:3,3] = translations
    transformations[:,3,3] = 1
    transformations = np.linalg.inv(transformations)
    np.save("tf0.npy", transformations)
    np.save("kp0.npy", kpt_arr)
else:
    kpt_arr = np.load("kp0.npy")
    transformations = np.load("tf0.npy")

rays = getRays(kpt_arr, transformations[:,0:3,0:3])
points, mask = getIntersectionPoints(rays, transformations[:,0:3,3])
'''
T = np.eye(4)
_,at,_ = cv2.estimateAffine3D(points, projpoints[ids])
T[:3, :] = at
T = np.linalg.inv(T)
'''
_,_,p = procrustes(points[mask], objpoints[mask], False)
tfp = np.eye(4)
tfp[:3,:3] = p['rotation']
tfp[:3, 3] = p['translation']
T = tfp

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
 
    tf = np.dot(tf, T)
    rvec,_ = cv2.Rodrigues(tf[:3, :3])
    tvec = tf[:3,3]
    imgpts,_ = cv2.projectPoints(projpoints, rvec, tvec, camera_matrix, None)
    imgpts = np.transpose(imgpts, (1,0,2))[0]
    
    labelfile = os.path.join(labelOutDir, 'label_' + repr(i).zfill(4) + '.txt')
    np.savetxt(labelfile, imgpts)

    if True:
        img = cv2.imread(os.path.join(rootPath, imagesDir, imgName + '.jpg'))
        for p in range(imgpts.shape[0]):
            cv2.circle(img, tuple((int(imgpts[p,0]), int(imgpts[p,1]))), 5, (0,0,255), -1)
        cv2.imshow("win", img)
        cv2.waitKey(200)
