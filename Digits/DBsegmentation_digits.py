import numpy as np
import cv2
import matplotlib
import os
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import glob
from PIL import Image
#blur_radius = 1

##greyscale
def average1(pixel):
    return ((299/1000)*pixel[0] + (587/1000)*pixel[1] + (114/1000)*pixel[2])/3
#L = R * 299/1000 + G * 587/1000 + B * 114/1000

def average2(pixel):
    return np.average(pixel);

# =============================================================================
# def binary(pixel):
#     if ((pixel[0] + pixel[1] + pixel[2])/3)
# =============================================================================


img = cv2.imread('test_images/32.png')
gray = cv2.imread('test_images/32.png')
img = ndimage.gaussian_filter(img, 0.1)
gray = ndimage.gaussian_filter(gray, 0.1)


print(img.shape)
w, h, d = img.shape

gray = np.dot(gray[...,:3],[0.299, 0.587, 0.114])
wg, hg = gray.shape

grey = np.zeros((img.shape[0],img.shape[1]))
bina = np.zeros((img.shape[0],img.shape[1]))

thresh = 25

row = len(img)

blackbackground = 0
if average1(img[1][1]) < thresh:
    blackbackground = 1
else:
    blackbackground = 0

if(blackbackground):
    for r in range(len(img)):
        for c in range(len(img[r])):
        # Use human average
           grey[r][c] = average1(img[r][c]);
           if grey[r][c] > thresh:
               grey[r][c] = 255
           else:
            grey[r][c] = 0
else:
    for r in range(len(img)):
        for c in range(len(img[r])):
    # Use human average
           grey[r][c] = average1(img[r][c]);
           if grey[r][c] > thresh:
               grey[r][c] = 0
           else:
               grey[r][c] = 255

print('Grey is this one')
# plt.imshow(grey)
# plt.show()

label = [0] * row
flag = 0


######crop the image#############
for r in range(len(img)):
    if label[r] == 1:
        continue
    for c in range(len(img[r])):
        #keep this row
        if grey[r][c] == 255:
            label[r] = 1
            #continue to next row
            break

grey2 = grey
index = 0
idx = []

for i in range(len(label)):
    if label[i] == 0:
        idx.append(index)
    index += 1

grey2 = np.delete(grey,idx, axis = 0)
##grey 2 = b&w, removing top and bottom empty rows

col = len(grey[0])

vertical_seg = []

n = 0

while(n < len(idx) - 1):
    if idx[n] != idx[n+1] - 1:
        vertical_seg.append(idx[n])
        vertical_seg.append(idx[n+1])
    index += 1
    n += 1
j = 0
z= -1
w = len(grey2[1])
verpath = 'Segment'
# #now vertical_seg has pairs of the y-axis coordinates for each line
# #crop each line and save each line
# #crop_img = img[y:y+h, x:x+w]
while(j < len(vertical_seg)):
    crop_img = img[ (vertical_seg[j]):(vertical_seg[j+1]),0:w]
    j = j + 2
    z = z + 1
    plt.imsave(os.path.join(verpath ,'Character' +str(z+1) + '.png'),crop_img, cmap = matplotlib.cm.Greys_r)


#iterate through x axis - each column
#stop when found the first 255
#find the next column where there is no 255 at all
#4 points - x coordinate of the first 255 and mix/max y, last 255 and mix/max y

##this is an array of the all zero columns

# =============================================================================
# npath = glob.glob("/Users/riachih/Desktop/Vertical/*.*")
# cv_img = []
# for img in npath:
#     n = cv2.imread(img)
#     cv_img.append(n)
#
#
# =============================================================================

# =============================================================================
# o = 0
# while(o < len(vertical_seg)):
#     h = (vertical_seg[o+1]) - (vertical_seg[o])
#     read_img =  np.zeros((h,img.shape[1]))
#     read_img = grey2[vertical_seg[o]:vertical_seg[o+1],:]
#     onlyZero = np.where(~read_img.any(axis=0))[0]
#     plt.imshow(read_img, cmap = matplotlib.cm.Greys_r)
#     onlyZero2 = onlyZero.tolist()
#     o = o + 2
# =============================================================================

##Test out for the cursive part
# =============================================================================
# img_row_sum = np.sum(grey2,axis=0).tolist()
# plt.plot(img_row_sum)
# plt.show()
#
# =============================================================================

image_list = []
for filename in glob.glob('Segment/*.*'): #assuming gif
    im=cv2.imread(filename)
    image_list.append(im)

b = 0

for file in image_list:
    grey = np.zeros((file.shape[0],file.shape[1]))
    row = len(file)

    for r in range(len(file)):
        for c in range(len(file[r])):
            # Use human average
            grey[r][c] = average1(file[r][c]);
            if grey[r][c] > 20:
                grey[r][c] = 0 #255
            else:
                grey[r][c] = 255 #0

    onlyZero = np.where(~grey.any(axis=0))[0]
    onlyZero2 = onlyZero.tolist()

##record each pair of non-zero columns
    nz_col = []
    index = 0
    temp1 = 0
    temp2 = 0
    check = 0
    i = 0
##iterate through onlyZero and find those that are not right next to each other
    while(i<(len(onlyZero2))):
        #if we have reached the end
        if i == len(onlyZero2) - 1:
            break;
        if onlyZero2[i] != onlyZero2[i+1] - 1:
            check = 1
            temp1 = onlyZero2[i]
            temp2 = onlyZero2[i+1]
            nz_col.append(temp1)
            nz_col.append(temp2)
            i = i+1
            continue
        else:
            i=i+1

    if len(nz_col)%2 == 1:
        nz_col.append(len(onlyZero2)-1)

##how many letters are inside
    letters = len(nz_col)//2
    h = len(grey)

    # plt.imshow(grey, cmap = matplotlib.cm.Greys_r)
    # plt.show()
    plt.imsave('test1.png', grey, cmap = matplotlib.cm.Greys_r)

    n = 0
    img = cv2.imread("test1.png")

    path = 'Result'
    a = 0

    #crop_img = img[y:y+h, x:x+w]
    while(n < len(nz_col)):
        crop_img = img[0:h, (nz_col[n]+1):(nz_col[n+1]-1)]
        n = n + 2
        # plt.imshow(crop_img, cmap = matplotlib.cm.Greys_r)
        # plt.show()
        labeled, nr_objects = ndimage.label(crop_img > 15)
        print ("Number of objects in cropped image is %d " % nr_objects)
        if nr_objects >1:
                temp = crop_img
                x = 0
                while x < nr_objects:
                    print(x)
    # #             #if x+1 == nr_objects:
    # #             #    break
                    loc = ndimage.find_objects(labeled)[x]
    # #            # loc2 = ndimage.find_objects(labeled)[x+1] #test
                    print(loc)
    # #            # print(loc2)
                    roi = crop_img[loc]
                    roi = cv2.bitwise_not(roi)
    # #            # roi2 = crop_img[loc2]
    # #            # for r in range(len(roi)):
    # #               #  for c in range(len(roi[r])):
    # #
                    # plt.imshow(roi)
                    plt.imsave(os.path.join(path ,'Character'+str(b)+ str(a+1) + '.png'),roi, cmap = matplotlib.cm.Greys_r)
                    x +=1
                    a+=1
        else:
                crop_img = cv2.bitwise_not(crop_img)
                plt.imsave(os.path.join(path ,'Character' +str(b) + str(a+1) + '.png'),crop_img, cmap = matplotlib.cm.Greys_r)
                a+=1
    b = b + 1
#
#
#
# =============================================================================
