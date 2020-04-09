# Max Filter
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images/butterfly.jpg', cv2.IMREAD_GRAYSCALE)
img_out = img.copy()
print(img.dtype)
height = img.shape[0]
width = img.shape[1]
#print('height=',height,'width = ',width)
kernelSize = 3

for i in np.arange(kernelSize, height-kernelSize):
    for j in np.arange(kernelSize, width-kernelSize):      
        max = 0
        for k in np.arange(-kernelSize, kernelSize+1):
            for l in np.arange(-kernelSize, kernelSize+1):
                a = img.item(i+k, j+l)
                if a > max:
                    max = a
        b = max
        img_out.itemset((i,j), b)
        
cv2.imwrite('images/filter_Max.jpg', img_out)
plt.figure()
plt.imshow(img,'gray')
plt.figure()
plt.imshow(img_out,'gray')
plt.show()





