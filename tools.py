import numpy as np
import cv2
from skimage import morphology
from skimage.filters import median
import matplotlib.pyplot as plt

import sys
import csv

import xml.etree.ElementTree as ET

import zipfile
import json
import time
import math
import io

def blobs_in_roi(blobs, roi):
    """Check if the center of blob is inside ROI  
    
    Arguments
    blobs -- list or array of areas occupied by the nanoparticle 
            (y, x, r) y and x are coordinates of the center and r - radius    
    roi -- (y,x,h,w)
    
    Return blobs list
    """
    indexes = list(map(lambda blob: int(blob[0]) >= roi[0] \
                                and int(blob[1]) >= roi[1] \
                                and int(blob[0]) < roi[0]+roi[2]  \
                                and int(blob[1]) < roi[1]+roi[3], \
                                    blobs))
    return np.copy(blobs[indexes]), indexes
# ----------------------------------------
def findIOU4circle(c1, c2):
    """Finds Jaccard similarity measure for two circles, 
       defined by the coordinates of centers and radii.
       c1=[x1,y1,r1], c2=[x2,y2,r2]  
    """

    d = np.linalg.norm(c1[:2] - c2[:2]) #distance betweem centers
        
    rad1sqr = c1[2] ** 2
    rad2sqr = c2[2] ** 2

    if d == 0:
        # the circle centers are the same
        return min(rad1sqr, rad2sqr)/max(rad1sqr, rad2sqr)

    angle1 = (rad1sqr + d ** 2 - rad2sqr) / (2 * c1[2] * d)
    angle2 = (rad2sqr + d ** 2 - rad1sqr) / (2 * c2[2] * d)

    # check if the circles are overlapping
    if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
        theta1 = np.arccos(angle1) * 2
        theta2 = np.arccos(angle2) * 2

        area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * np.sin(theta2))
        area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * np.sin(theta1))

        return (area1 + area2)/(np.pi*(rad1sqr+rad2sqr) - area1 - area2)

    elif angle1 < -1 or angle2 < -1:
        # Smaller circle is completely inside the largest circle.
        # Intersection area will be area of smaller circle
        # return area(c1_r), area(c2_r)
        return min(rad1sqr, rad2sqr)/max(rad1sqr, rad2sqr)
    return 0

def accur_estimation2(blobs_gt, blobs_est, roi, thres=0.5):
    
    blobs_gt, _ = blobs_in_roi(blobs_gt, roi)
    blobs_est, _ = blobs_in_roi(blobs_est, roi)  

    length_gt = blobs_gt.shape[0]
    length_est = blobs_est.shape[0]
        
    iou = np.zeros((length_gt, length_est))
    for i in range(length_gt):
        for j in range(length_est):
            iou[i,j] = findIOU4circle(blobs_gt[i], blobs_est[j])
    
    match = 0
    no_match = 0
    fake = 0
    no_match_index = np.zeros(length_gt,dtype = 'bool')
    match_index = np.zeros(length_gt, dtype='bool')
    
    match_matr = np.zeros((length_gt, length_est), dtype = int)

    for i in range(length_gt):
        if max(iou[i])>=thres:
            imax = np.argmax(iou[i])
            match_matr[i,imax] = 1
            
    no_match_gt_blobs =  blobs_gt[no_match_index]    
    
    fake_index = np.zeros(length_est,dtype = 'bool')
    truedetected_blobs_index = np.zeros(length_est,dtype = 'bool')
    for j in range(length_est):
        if sum(match_matr[:,j])>1: 
            imax = np.argmax(iou[:,j])
            match_matr[:, j] = np.zeros(length_gt, dtype = int)
            match_matr[imax, j] = 1 
        if sum(match_matr[:,j]) == 0:
            fake+=1
            fake_index[j] = True
        else:
            truedetected_blobs_index[j] = True
    fake_blobs = blobs_est[fake_index]
        
    for i in range(length_gt): 
        if sum(match_matr[i,:]) == 0: 
            no_match_index[i] = True
        else:
            match_index[i] = True

    no_match = sum(no_match_index)
    match = sum(sum(match_matr))        
    no_match_gt_blobs =  blobs_gt[no_match_index]
    match_blobs = blobs_gt[match_index]
    truedetected_blobs = blobs_est[truedetected_blobs_index]
    
    return match, no_match, fake, no_match_gt_blobs, fake_blobs, match_blobs, truedetected_blobs

def blobs2roi(_blobs, _heightImg, _widthImg):
    roi = np.zeros(4, dtype='int')
    roi[0] = max(0, (_blobs[:,0]-_blobs[:,2]).min()) 
    roi[1] = max(0, (_blobs[:,1]-_blobs[:,2]).min())
    roi[2] = min(_heightImg, (_blobs[:,0]+_blobs[:,2]).max() - roi[0]+1)
    roi[3] = min(_widthImg, (_blobs[:,1]+_blobs[:,2]).max() - roi[1]+1)
    return roi 

def showDiff(temp_img, roi, blobs_est, gt_blobs, no_match_gt_blobs, fake_blobs, ax):
    data2show = {'results':1, 'ground truth':1, 'missed gt':1, 'fake est':1}
    color = ['blue', 'lime', 'red', 'yellow'] #ground truth, calculated, not found, fake

    ax.imshow(temp_img, cmap='gray')

    if data2show['results']:
        for blob in blobs_est :
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color[0], linewidth=0.5, fill=False)
            ax.add_patch(c)

    if data2show['ground truth']:
        for blob in gt_blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color[1], linewidth=0.5, fill=False)
            ax.add_patch(c)

    if data2show['missed gt']:
        for blob in no_match_gt_blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color[2], linewidth=0.5, fill=False)
            ax.add_patch(c)    

    if data2show['fake est']:
        for i, blob in enumerate(fake_blobs):
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color[3], linewidth=0.75, fill=False)
            ax.add_patch(c)               
            #plt.text(x+3, y+3, str(i), fontsize=7, color = 'pink')

    ax.set_axis_off()
    plt.tight_layout()

def ImportTaskFromCVAT(taskCVAT):
    with zipfile.ZipFile(taskCVAT, 'r') as tempZipFile:
        annotations = tempZipFile.read('annotations.json')
        annotations = annotations.decode('utf-8')

        manifest = tempZipFile.read('data/manifest.jsonl')
        manifest = manifest.decode('utf-8')
        temp = json.loads(manifest.split('\n')[-1])
        imgFileName = temp['name'] + temp['extension']        
        imageBytes = tempZipFile.read(f'data/{imgFileName}')
    
    BLOBs = []

    annotations = json.loads(annotations)
    for shape in annotations[0]['shapes']:
        points = shape['points']

        coordinates = [[points[0], points[1]], [points[2], points[3]]]

        d = math.dist(coordinates[0], coordinates[1])
        x = (coordinates[0][0] + coordinates[1][0]) / 2
        y = (coordinates[0][1] + coordinates[1][1]) / 2
        
        BLOBs.append([y, x, d])

    return np.array(BLOBs), imgFileName, io.BytesIO(imageBytes)