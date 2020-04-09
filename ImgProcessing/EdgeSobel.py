


# Edge detection using Sobel Operator
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
# Convolution between two matrix X and F
def convolve_np(X, F):
    X_height,X_width = X.shape
    F_height,F_width = F.shape

    H = np.int64((F_height - 1) / 2)
    W = np.int64((F_width - 1) / 2)
    
    out = np.zeros((X_height, X_width))
    
    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = 0
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    a = X[i+k, j+l]
                    w = F[H+k, W+l]
                    sum += (w * a)
            out[i,j] = sum

    return out

img = cv2.imread('images/butterfly.jpg', cv2.IMREAD_GRAYSCALE)
img_out = img.copy()
print(img.dtype)
height = img.shape[0]
width = img.shape[1]

Hx,Hy = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]), np.array([[-1, -2, -1],
                                          [ 0,  0,  0],
                                          [ 1,  2,  1]])  # Sobel kernels
t0 = time.time()
img_x = convolve_np(img, Hx)   # gradient in x-direction
img_y = convolve_np(img, Hy)   # gradient in y-direction
img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))
img_out = (img_out / np.max(img_out)) * 255   # normalize to [0-255]
t1 = time.time()
print(t1-t0)

cv2.imwrite('images/edge_sobel.jpg', img_out)
plt.imshow(img_out, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()








