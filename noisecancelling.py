import numpy as np
import cv2
import matplotlib
import os
import matplotlib.pyplot as plt 
import scipy
from scipy import ndimage
import glob
from PIL import Image
import math
#blur_radius = 1

##greyscale
def average1(pixel):
    return ((299/1000)*pixel[0] + (587/1000)*pixel[1] + (114/1000)*pixel[2])
#L = R * 299/1000 + G * 587/1000 + B * 114/1000

def average2(pixel):
    return np.average(pixel);

# =============================================================================
# def binary(pixel):
#     if ((pixel[0] + pixel[1] + pixel[2])/3) 
# =============================================================================


img = cv2.imread('WechatIMG80.jpeg')
gray = cv2.imread('WechatIMG80.jpeg')
img = ndimage.gaussian_filter(img, 0.1)
gray = ndimage.gaussian_filter(gray, 0.1)


print(img.shape)
width, height, d = img.shape

##breaks into 10 horizontal-blocks
hor = round(width/10)
#width

#numbers of vertical blocks
ver_num = round(10 * height/width)
ver = height / ver_num
#hor
num_pix = round(hor*ver)

gray = np.dot(gray[...,:3],[0.299, 0.587, 0.114])
wg, hg = gray.shape

grey = np.zeros((img.shape[0],img.shape[1]))
bina = np.zeros((img.shape[0],img.shape[1]))

#turns into greyscale first
for r in range(len(img)): 
    for c in range(len(img[r])): 
        # Use human average
        grey[r][c] = average1(img[r][c]);

hor_count = 1
ver_count = 1
        
thresh = 33

row = len(img)

#going through the blocks
hor_count = 0
#10 in total

ver_count = 0
#ver_num = 10 * height/width in total
status = 0
all_background = 0
all_foreground = 0
bina = 0
record = np.zeros((round(ver_num),10))
threshold = []

while(hor_count < 10):
    ver_count = 0
    while(ver_count < round(ver_num)):
        temp = grey[round(ver_count*ver):round((ver_count+1)*ver), round(hor_count*hor): round((hor_count+1)*hor)]
        std = np.std(temp)
        mean = np.mean(temp)
        if std <= 15:
            if mean < 130:
                status = 1
                #all background
                record[ver_count,hor_count] = 1
            else:
                record[ver_count,hor_count] = 2
        else:
            record[ver_count,hor_count] = 3
            threshold.append(0.875*mean)
            
            
        ver_count = ver_count + 1
    hor_count = hor_count + 1

ver_count = 0
hor_count = 0
j = 0;
for r in range(len(record)): 
    for c in range(len(record[r])):
        if record[r][c] == 1:
            for x in range(int(r*ver),int((r+1)*ver)-1):
                for y in range(int(c*hor), int((c+1)*hor)-1):
                  grey[x][y] = 0;
        elif record[r][c] == 2:
            for x in range(int(r*ver), int((r+1)*ver)-1):
                for y in range(int(c*hor), int((c+1)*hor)-1):
                  grey[x][y] = 255;
        else:
            for x in range(int(r*ver), int((r+1)*ver)-1):
                for y in range(int(c*hor), int((c+1)*hor)-1):
                    if grey[x][y] < threshold[j]:
                        grey[x][y] = 0
                    else:
                        grey[x][y] = 255
            j = j+1
                
            
# =============================================================================
# blackbackground = 0
# if average1(img[1][1]) < thresh:
#     blackbackground = 1
# else:
#     blackbackground = 0
# 
# if(blackbackground):
#     for r in range(len(img)): 
#         for c in range(len(img[r])): 
#         # Use human average
#            grey[r][c] = average1(img[r][c]);
#            if grey[r][c] > thresh:
#                grey[r][c] = 255
#            else:
#             grey[r][c] = 0
# else:
#     for r in range(len(img)): 
#         for c in range(len(img[r])): 
#     # Use human average
#            grey[r][c] = average1(img[r][c]);
#            if grey[r][c] > thresh:
#                grey[r][c] = 0
#            else:
#                grey[r][c] = 255
# =============================================================================

print('Grey is this one')
plt.imshow(grey)
plt.show()



