#coding:utf-8
import cv2
assert cv2.__version__[0] == '3'
import numpy as np
import os
import glob

def get_K_and_D(checkerboard, imgsPath):

    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] 
    imgpoints = [] 
    images = glob.glob(imgsPath + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            print(fname)
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, _img_shape[::-1], None, None)
  
    #rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints,imgpoints,gray.shape[::-1],K,D,rvecs,tvecs,calibration_flags,
                # (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    
    return DIM, mtx, dist

def undistort(img):
    K = np.array([[1594.7850171788803, 0.0, 1068.9712535174892], [0.0, 1601.5045483878248, 558.9450538452985], [0.0, 0.0, 1.0]])
    D = np.array([[-0.18354063869482787], [1.5301533081939518], [-7.575119029711882], [13.057055776523992]])
    DIM = (2200, 1300)
    #img = cv2.imread(img_path)
    #img = cv2.resize(img, DIM)
    #image = np.zeros((1160, 2100, 3),dtype=np.uint8)
    #for i in range(40, 1120):
    #    for j in range(90, 2010):
    #        #print(i, j)
    #        image[i,j,:]=image[i,j,:] + img[i-40, j-90,:]
    #image = image.astype(np.uint8)
    #print(image.shape)
    #cv2.imshow('im', image)
    #cv2.waitKey(0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    print(map1.shape)
    print(map2.shape)
    np.save('map1.npy', map1)
    np.save("map2.npy", map2)
    '''
    f = open('map1.txt', 'w')
    for i in range(map1.shape[0]):
        for j in range(map1.shape[1]):
            m = map1[i, j]
            f.write(str(m[0])+' '+str(m[1])+'\n')
    f.close()
 
    f = open('map2.txt', 'w')
    for i in range(map2.shape[0]):
        line = ''
        for j in range(map2.shape[1]):
            line += str(map2[i, j]) + ' '
        f.write(line + '\n')
    f.close()
    '''

    #undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)    
    #cv2.imwrite('unfisheyeImage1.jpg', undistorted_img)
    return undistorted_img

def undistort_test(img_path, DIM, K, D):
    img = cv2.imread(img_path)
    img = cv2.resize(img, DIM)
    dst = cv2.undistort(img, K, D, None, K)
    #map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    #undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    cv2.imwrite('unfisheyeImage.jpg', dst)
    return dst

if __name__ == '__main__':
    #DIM, K, D = get_K_and_D((6,9), 'fisheye')
    #print(K,D,DIM)
    
    im = undistort('67.jpg')
    #undistorted_img = undistort_test('16.jpg', DIM, K, D)
