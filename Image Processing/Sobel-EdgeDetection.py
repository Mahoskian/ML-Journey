import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def my_convolve2d(img, kernel, mode='same'):
    k_rows, k_cols = kernel.shape
    i_rows, i_cols = img.shape
    pad_h = (k_rows - 1) // 2
    pad_w = (k_cols - 1) // 2
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    out = np.zeros((i_rows, i_cols))

    # loop unrolling
    for i in range(-pad_h, pad_h+1):
        for j in range(-pad_w, pad_w+1):
            # compute the convolution for this kernel element
            kernel_element = kernel[i+pad_h, j+pad_w]
            img_slice = img_padded[pad_h+i:i_rows+pad_h+i, pad_w+j:i_cols+pad_w+j]
            out += img_slice * kernel_element

    return out

# Define a custom colormap that maps values from 0 to 1 to grayscale
gray_map = plt.cm.colors.LinearSegmentedColormap.from_list('my_gray', [(0, 'black'), (1, 'white')])

# Load image
img = plt.imread(r'C:\Users\soham\Documents\GitHub\ML-Journey\Image Processing\input-images\image.jpg')

# Convert to grayscale
gray = np.dot(img[..., :3], [1.0, 1.0, 1.0])

# Pre-Defined Sobel kernels
kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Compute gradients
grad_x = np.abs(convolve2d(gray, kernel_x, mode='same'))
grad_y = np.abs(convolve2d(gray, kernel_y, mode='same'))
grad = np.sqrt(grad_x**2 + grad_y**2)

# Normalize values to between 0 and 1, subtract from 1 to swap balck/white
grad = 1 - (grad / np.max(grad))

# Show edge detection result
plt.imsave(r'C:\Users\soham\Documents\GitHub\ML-Journey\Image Processing\output-images\Sobel-Edge-Detection-Result.png', grad, cmap=gray_map)
plt.imshow(grad, cmap=gray_map)
plt.show()