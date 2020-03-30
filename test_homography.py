#!/usr/bin/python3

import matplotlib.pyplot as plt
import cv2
from homography import *

img = cv2.cvtColor(cv2.imread('test_image.jpg'), cv2.COLOR_BGR2RGB)
img_shape = img.shape
src = [[90, 600],[332, 325],[img_shape[1]-332, 325],[img_shape[1], 600]]
dst = [[200, img_shape[0]],[200, 250],[img_shape[1]-200, 250],[img_shape[1]-200, img_shape[0]]]
H, H_inverse = homography(src,dst)

test_input = [img_shape[1]-440, 190]

warped_img = np.zeros(img_shape, dtype=np.int32)
for row in range(img_shape[0]):
    for col in range(img_shape[1]):
        # For each destination point compute corresponding a source point
        xy_src = applyHomography([col, row], H_inverse)
        if ((xy_src[0] >= 0) and (xy_src[0] < img_shape[1])) and    \
            ((xy_src[1] >= 0) and (xy_src[1] < img_shape[0])):
            warped_img[row][col] = img[xy_src[1]][xy_src[0]]

plt.subplot(2,1,1)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(2,1,2)
plt.imshow(warped_img)
plt.title('Warped Image')
plt.show()
